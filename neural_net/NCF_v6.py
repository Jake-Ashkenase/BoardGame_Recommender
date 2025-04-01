import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def silence_worker_output(worker_id):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


##############################################
# Helper: Recursively move data to device
##############################################
def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    else:
        return data


##############################################
# Feature Grouping Functions
##############################################
def get_game_feature_groups(df):
    text_cols = [col for col in df.columns if col.startswith("desc_emb_")]
    summary_cols = [col for col in df.columns if col.startswith("user_rating") or
                    col.startswith("prop_") or col == "total_ratings"]
    info_candidates = ["YearPublished", "GameWeight", "AvgRating", "BayesAvgRating", "StdDev",
                       "MinPlayers", "MaxPlayers", "ComAgeRec", "LanguageEase", "NumOwned", "NumWant",
                       "NumWish", "NumWeightVotes", "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime",
                       "MfgAgeRec", "NumUserRatings", "NumAlternates", "NumExpansions", "NumImplementations",
                       "IsReimplementation", "Kickstarted", "Rank:boardgame", "Rank:strategygames", "Rank:abstracts",
                       "Rank:familygames", "Rank:thematic", "Rank:cgs", "Rank:wargames", "Rank:partygames",
                       "Rank:childrensgames"]
    info_cols = [col for col in df.columns if col in info_candidates]
    used_cols = set(text_cols + summary_cols + info_cols + ["BGGId"])
    try:
        low_exp_designer_idx = df.columns.get_loc("Low-Exp Designer")
    except KeyError:
        low_exp_designer_idx = None
    if low_exp_designer_idx is not None:
        binary_cols = [col for col in df.columns if col not in used_cols and
                       df.columns.get_loc(col) < low_exp_designer_idx]
    else:
        binary_cols = []
    if low_exp_designer_idx is not None:
        if binary_cols:
            start_designers_idx = df.columns.get_loc(binary_cols[-1]) + 1
        else:
            last_info_idx = max([df.columns.get_loc(col) for col in info_cols]) if info_cols else -1
            start_designers_idx = last_info_idx + 1
        designers_cols = list(df.columns[start_designers_idx: low_exp_designer_idx + 1])
    else:
        designers_cols = []
    try:
        low_exp_publisher_idx = df.columns.get_loc("Low-Exp Publisher")
    except KeyError:
        low_exp_publisher_idx = None
    if low_exp_publisher_idx is not None:
        if low_exp_designer_idx is not None:
            start_publishers_idx = low_exp_designer_idx + 1
        else:
            start_publishers_idx = df.columns.get_loc(binary_cols[-1]) + 1 if binary_cols else 0
        publishers_cols = list(df.columns[start_publishers_idx: low_exp_publisher_idx + 1])
    else:
        publishers_cols = []
    groups = {
        "text": text_cols,
        "summary": summary_cols,
        "info": info_cols,
        "binary": binary_cols,
        "designers": designers_cols,
        "publishers": publishers_cols
    }
    return groups


def build_game_features_map(games_df, groups):
    game_features_map = {}
    for _, row in games_df.iterrows():
        bggid = row["BGGId"]
        group_features = {}
        for key, cols in groups.items():
            if cols:
                vals = row[cols].values.astype(np.float32)
                group_features[key] = torch.tensor(vals, dtype=torch.float32)
        game_features_map[bggid] = group_features
    return game_features_map


def collate_features(feature_list):
    return {key: torch.stack([feat.get(key) for feat in feature_list])
            for key in feature_list[0].keys()}


##############################################
# Dataset Class
##############################################
class RatingsDatasetMulti(Dataset):
    def __init__(self, ratings_df, game_features_map, user_id_map, game_id_map):
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.game_features_map = game_features_map
        self.user_id_map = user_id_map
        self.game_id_map = game_id_map
        if torch.utils.data.get_worker_info() is not None:
            sys.stdout = open(os.devnull, 'w')

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_id = row['Username']
        bggid = row['BGGId']
        rating = row['NormalizedRating'] if 'NormalizedRating' in row else row['Rating']
        user_index = self.user_id_map[user_id]
        game_index = self.game_id_map[bggid]
        features = self.game_features_map.get(bggid, {})
        return features, torch.tensor(user_index, dtype=torch.long), torch.tensor(rating,
                                                                                  dtype=torch.float32), torch.tensor(
            game_index, dtype=torch.long)


##############################################
# Multi-Branch Game Encoder using ModuleDict
##############################################
class MultiBranchGameEncoder(nn.Module):
    def __init__(self, groups_dims, fusion_out_dim=256,
                 branch_configs=None, designer_out_dim=32, publisher_out_dim=32):
        """
        groups_dims: dict mapping group names to input dimensions.
        branch_configs: dict mapping group names to (hidden_layers, final_output_dim).
        For designers/publishers, groups_dims are the number of binary flags.
        """
        super(MultiBranchGameEncoder, self).__init__()
        if branch_configs is None:
            branch_configs = {
                "text": ([128, 64], 64),
                "summary": ([32], 32),
                "info": ([32], 32),
                "binary": ([64], 64)
            }
        self.branches = nn.ModuleDict()
        self.out_dims = {}
        # Use moderate dropout (0.2) here.
        for key in ["text", "summary", "info", "binary"]:
            if groups_dims.get(key, 0) > 0:
                layers = []
                in_dim = groups_dims[key]
                for hidden in branch_configs[key][0]:
                    layers.append(nn.Linear(in_dim, hidden))
                    layers.append(nn.BatchNorm1d(hidden))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    in_dim = hidden
                self.branches[key] = nn.Sequential(*layers)
                self.out_dims[key] = branch_configs[key][1]
            else:
                self.out_dims[key] = 0
        if groups_dims.get("designers", 0) > 0:
            self.num_designers = groups_dims["designers"]
            self.designer_emb = nn.Parameter(torch.randn(self.num_designers, designer_out_dim))
            self.out_dims["designers"] = designer_out_dim
        else:
            self.out_dims["designers"] = 0
        if groups_dims.get("publishers", 0) > 0:
            self.num_publishers = groups_dims["publishers"]
            self.publisher_emb = nn.Parameter(torch.randn(self.num_publishers, publisher_out_dim))
            self.out_dims["publishers"] = publisher_out_dim
        else:
            self.out_dims["publishers"] = 0
        total_in = sum(self.out_dims.values())
        # Fusion network with 0.3 dropout.
        self.fusion = nn.Sequential(
            nn.Linear(total_in, total_in * 2),
            nn.BatchNorm1d(total_in * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_in * 2, fusion_out_dim)
        )

    def forward(self, features_dict):
        outs = []
        for key, branch in self.branches.items():
            if key in features_dict:
                outs.append(branch(features_dict[key]))
        if "designers" in features_dict:
            x = features_dict["designers"]
            designer_sum = torch.matmul(x, self.designer_emb)
            counts = x.sum(dim=1, keepdim=True) + 1e-6
            outs.append(designer_sum / counts)
        if "publishers" in features_dict:
            x = features_dict["publishers"]
            publisher_sum = torch.matmul(x, self.publisher_emb)
            counts = x.sum(dim=1, keepdim=True) + 1e-6
            outs.append(publisher_sum / counts)
        fused = torch.cat(outs, dim=1)
        return self.fusion(fused)


##############################################
# Pairwise Ranking Loss Function
##############################################
def pairwise_ranking_loss(mu, y, margin=0.1):
    """
    Computes a pairwise ranking loss on predicted means.
    For each pair (i,j) in the batch where y[i] > y[j],
    encourages mu[i] - mu[j] to be greater than margin.
    """
    diff_mu = mu.unsqueeze(1) - mu.unsqueeze(0)  # (B, B)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)  # (B, B)
    valid = diff_y != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=mu.device)
    target = torch.sign(diff_y[valid])
    loss = torch.clamp(margin - target * diff_mu[valid], min=0)
    return loss.mean()


##############################################
# Combined Loss: Heteroscedastic Loss + Ranking Loss
##############################################
class CombinedLoss(nn.Module):
    def __init__(self, hetero_loss, ranking_loss_weight=0.1, margin=0.1):
        super(CombinedLoss, self).__init__()
        self.hetero_loss = hetero_loss
        self.ranking_loss_weight = ranking_loss_weight
        self.margin = margin

    def forward(self, predictions, targets):
        hetero = self.hetero_loss(predictions, targets)
        mu = predictions[:, 0]
        ranking = pairwise_ranking_loss(mu, targets, margin=self.margin)
        return hetero + self.ranking_loss_weight * ranking


##############################################
# Heteroscedastic Loss Function (unchanged)
##############################################
class HeteroscedasticLoss(nn.Module):
    def __init__(self, rating_to_weight=None, min_rating=0.0, max_rating=10.0, precision=2):
        super(HeteroscedasticLoss, self).__init__()
        self.rating_to_weight = rating_to_weight
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.precision = precision
        if rating_to_weight is not None:
            scale = 10 ** precision
            num_bins = int(round((max_rating - min_rating) * scale)) + 1
            weight_list = []
            for i in range(num_bins):
                rating_val = min_rating + i / scale
                weight = rating_to_weight.get(round(rating_val, precision), 1.0)
                weight_list.append(weight)
            self.register_buffer('weight_tensor', torch.tensor(weight_list, dtype=torch.float32))
            self.scale = scale
        else:
            self.weight_tensor = None

    def forward(self, predictions, targets):
        mu = predictions[:, 0]
        log_var = predictions[:, 1]
        loss = 0.5 * torch.exp(-log_var) * (targets - mu) ** 2 + 0.5 * log_var
        if self.weight_tensor is not None:
            indices = torch.round((targets - self.min_rating) * self.scale).long()
            indices = torch.clamp(indices, 0, self.weight_tensor.numel() - 1)
            weights = self.weight_tensor[indices]
            loss = weights * loss
        return torch.mean(loss)


##############################################
# Rating Predictor with Heteroscedastic Regression and Variance Clipping
##############################################
class RatingPredictor(nn.Module):
    def __init__(self, game_encoder, num_users, num_games, latent_dim=256,
                 user_emb_dim=256, mlp_hidden_dim=1024, dropout_rate=0.3):
        super(RatingPredictor, self).__init__()
        self.game_encoder = game_encoder
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.game_embedding = nn.Embedding(num_games, latent_dim)
        self.game_bias = nn.Embedding(num_games, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        input_size = latent_dim + user_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.BatchNorm1d(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
            nn.BatchNorm1d(mlp_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 4, 2)  # [mu, log_var]
        )

    def forward(self, game_features_dict, user_ids, game_ids):
        #game_emb = self.game_encoder(game_features_dict)
        game_emb = self.game_embedding(game_ids)
        user_emb = self.user_embedding(user_ids)
        bias = self.game_bias(game_ids).squeeze(1)
        mlp_input = torch.cat([game_emb, user_emb], dim=1)
        out = self.mlp(mlp_input)
        mu = out[:, 0] + bias + self.global_bias
        log_var = torch.clamp(out[:, 1], min=-5, max=3)
        return torch.stack([mu, log_var], dim=1)


##############################################
# Unified Training/Evaluation Functions with Early Stopping
##############################################
def process_epoch(model, loader, optimizer, criterion, device, train=False):
    model.train() if train else model.eval()
    total_loss = 0.0
    for batch in loader:
        features_dict, user_ids, ratings, game_ids = batch
        features_dict = to_device(features_dict, device)
        user_ids = user_ids.to(device)
        ratings = ratings.to(device)
        game_ids = game_ids.to(device)
        outputs = model(features_dict, user_ids, game_ids)
        loss = criterion(outputs, ratings)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * user_ids.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, device='cpu', patience=3):
    train_losses = []
    test_losses = []
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        tr_loss = process_epoch(model, train_loader, optimizer, criterion, device, train=True)
        te_loss = process_epoch(model, test_loader, optimizer, criterion, device, train=False)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {tr_loss:.4f}, Test Loss: {te_loss:.4f}")
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        if te_loss < best_loss:
            best_loss = te_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "ncf_simplified.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    return train_losses, test_losses


def get_true_and_pred(model, loader, device, mu, sigma):
    model.eval()
    true_ratings = []
    predicted_ratings = []
    with torch.no_grad():
        for batch in loader:
            features_dict, user_ids, ratings, game_ids = batch
            features_dict = to_device(features_dict, device)
            user_ids = user_ids.to(device)
            ratings = ratings.to(device)
            game_ids = game_ids.to(device)
            outputs = model(features_dict, user_ids, game_ids)
            mu_pred = outputs[:, 0]
            true_ratings.extend(ratings.cpu().numpy())
            predicted_ratings.extend(mu_pred.cpu().numpy())
    true_ratings = np.array(true_ratings) * sigma + mu
    predicted_ratings = np.array(predicted_ratings) * sigma + mu
    return true_ratings, predicted_ratings


##############################################
# Plotting Functions
##############################################
def plot_losses(train_losses, test_losses, title="Training and Test Loss Over Epochs"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_pred(true, pred, title="True Ratings vs. Predicted Ratings"):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, alpha=0.5, label="Predictions")
    mval = min(true.min(), pred.min())
    Mval = max(true.max(), pred.max())
    plt.plot([mval, Mval], [mval, Mval], 'r--', label="Perfect Prediction")
    plt.xlabel("True Ratings (Original Scale)")
    plt.ylabel("Predicted Ratings (Original Scale)")
    plt.title(title)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(true, pred, save_as=None, aggregate=True, shaded_region=True, smoothing_window=5,
                     title="Predicted vs. Actual"):
    plt.figure(figsize=(8, 6), facecolor='#384957')
    if aggregate or shaded_region:
        df = pd.DataFrame({"actual": true, "predicted": pred})
        grouped = df.groupby("actual")["predicted"]
        mean_pred = grouped.mean()
        min_pred = grouped.min()
        max_pred = grouped.max()
        sorted_actual = mean_pred.index
        smoothed_mean = mean_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smoothed_min = min_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smoothed_max = max_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        if shaded_region:
            plt.fill_between(sorted_actual, smoothed_min.values, smoothed_max.values,
                             color="#FF6B65", alpha=0.15, label="Prediction Range")
        if aggregate:
            plt.plot(sorted_actual, smoothed_mean.values, color="#FF6B65", label="Mean Prediction")
    else:
        plt.scatter(true, pred, alpha=0.5, label="Predicted vs Actual", color="#FF6B65", s=5)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--', label="Perfect Prediction", color="#384957")
    plt.xlabel("Actual", color="white")
    plt.ylabel("Predicted", color="white")
    plt.title(title, color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
    plt.show()


##############################################
# Collate Function for DataLoader
##############################################
def collate_fn(batch):
    features_list, user_ids, ratings, game_ids = zip(*batch)
    return collate_features(features_list), torch.stack(user_ids), torch.stack(ratings), torch.stack(game_ids)


##############################################
# Preprocessing: Train/Test Split and Weights
##############################################
def preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42):
    user_counts = ratings_df['Username'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    ratings_df = ratings_df[ratings_df['Username'].isin(valid_users)]
    game_counts = ratings_df['BGGId'].value_counts()
    valid_games = game_counts[game_counts >= 2].index
    ratings_df = ratings_df[ratings_df['BGGId'].isin(valid_games)]
    train_list, test_list = [], []
    rng = np.random.RandomState(random_state)
    for user, group in ratings_df.groupby('Username'):
        group = group.sample(frac=1, random_state=rng.randint(0, 10000))
        num_ratings = len(group)
        num_test = max(1, int(np.floor(num_ratings * test_size)))
        test_ratings = group.iloc[:num_test]
        train_ratings = group.iloc[num_test:]
        if train_ratings.empty:
            train_ratings = test_ratings.iloc[[0]]
            test_ratings = test_ratings.iloc[1:]
        train_list.append(train_ratings)
        test_list.append(test_ratings)
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    all_games = set(ratings_df['BGGId'].unique())
    games_in_train = set(train_df['BGGId'].unique())
    games_in_test = set(test_df['BGGId'].unique())
    missing_in_train = all_games - games_in_train
    missing_in_test = all_games - games_in_test
    for game in missing_in_train:
        candidate_rows = test_df[test_df['BGGId'] == game]
        if not candidate_rows.empty:
            row_to_move = candidate_rows.iloc[[0]]
            train_df = pd.concat([train_df, row_to_move], ignore_index=True)
            test_df = test_df.drop(candidate_rows.index[0])
    for game in missing_in_test:
        candidate_rows = train_df[train_df['BGGId'] == game]
        if not candidate_rows.empty:
            row_to_move = candidate_rows.iloc[[0]]
            test_df = pd.concat([test_df, row_to_move], ignore_index=True)
            train_df = train_df.drop(candidate_rows.index[0])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_inverse_frequency_weights_normalized(ratings_df):
    rating_counts = ratings_df['NormalizedRating'].value_counts().to_dict()
    return {round(rating, 2): 1.0 / count for rating, count in rating_counts.items()}


##############################################
# Main Training Script
##############################################
def main():
    # Load game data.
    games_df = pd.read_csv("../../bgg_data/overall_games.csv")
    games_df = games_df.drop(columns=["Name"], errors='ignore')

    unique_game_ids = games_df["BGGId"].unique()
    game_id_map = {bgid: idx for idx, bgid in enumerate(unique_game_ids)}

    # Load ratings data.
    ratings_df = pd.read_csv("../../bgg_data/ratings_filtered.csv")
    ratings_df = ratings_df[ratings_df["BGGId"].isin(set(games_df["BGGId"]))]
    unique_users = ratings_df['Username'].unique()
    # Use all users (or a subset if desired).
    ratings_df = ratings_df[ratings_df["Username"].isin(unique_users)]
    num_users = len(unique_users)
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}

    print(f"Total ratings: {len(ratings_df)}")
    train_ratings_df, test_ratings_df = preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42)
    print("Train-test split done.")

    mu_rating = train_ratings_df['Rating'].mean()
    sigma_rating = train_ratings_df['Rating'].std()
    print(f"Rating mean: {mu_rating:.2f}, std: {sigma_rating:.2f}")

    train_ratings_df['NormalizedRating'] = ((train_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)
    test_ratings_df['NormalizedRating'] = ((test_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)

    rating_to_weight = compute_inverse_frequency_weights_normalized(train_ratings_df)

    groups = get_game_feature_groups(games_df)
    game_features_map = build_game_features_map(games_df, groups)
    groups_dims = {key: len(val) for key, val in groups.items()}

    train_dataset = RatingsDatasetMulti(train_ratings_df, game_features_map, user_id_map, game_id_map)
    test_dataset = RatingsDatasetMulti(test_ratings_df, game_features_map, user_id_map, game_id_map)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              worker_init_fn=silence_worker_output, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                             worker_init_fn=silence_worker_output, collate_fn=collate_fn)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    latent_dim = 256
    game_encoder = MultiBranchGameEncoder(groups_dims, fusion_out_dim=latent_dim)
    num_games = len(unique_game_ids)
    model = RatingPredictor(game_encoder, num_users, num_games, latent_dim=latent_dim,
                            user_emb_dim=256, mlp_hidden_dim=1024)
    # Optionally load pretrained weights:
    # state_dict = torch.load("ncf_model_hetero.pth")
    # model.load_state_dict(state_dict)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # Normalized ratings are roughly in [-3,3] if mu is around the global mean.
    base_hetero_loss = HeteroscedasticLoss(rating_to_weight, min_rating=-3.0, max_rating=3.0, precision=2)
    # Combine with ranking loss (with margin=0.1 and ranking weight 0.1, adjust as needed).
    criterion = CombinedLoss(base_hetero_loss, ranking_loss_weight=0.1, margin=0.1).to(device)

    num_epochs = 15
    patience = 3
    print("Starting training...")
    train_losses, test_losses = train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion,
                                            device=device, patience=patience)

    plot_losses(train_losses, test_losses)

    true_test, pred_test = get_true_and_pred(model, test_loader, device, mu_rating, sigma_rating)
    plot_predictions(true_test, pred_test, title="Test Data: True vs. Predicted Ratings", smoothing_window=10)

    true_train, pred_train = get_true_and_pred(model, train_loader, device, mu_rating, sigma_rating)
    plot_predictions(true_train, pred_train, title="Train Data: True vs. Predicted Ratings", smoothing_window=10)


if __name__ == "__main__":
    main()
