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
# Generic helper to recursively move data to device.
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
    # Group 1: Text embeddings
    text_cols = [col for col in df.columns if col.startswith("desc_emb_")]

    # Group 2: Rating summaries: any column starting with "user_rating", "prop_" or "total_ratings"
    summary_cols = [col for col in df.columns if
                    col.startswith("user_rating") or col.startswith("prop_") or col == "total_ratings"]

    # Group 3: Basic game info (manually specified)
    info_candidates = ["YearPublished", "GameWeight", "AvgRating", "BayesAvgRating", "StdDev",
                       "MinPlayers", "MaxPlayers", "ComAgeRec", "LanguageEase", "NumOwned", "NumWant",
                       "NumWish", "NumWeightVotes", "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime",
                       "MfgAgeRec", "NumUserRatings", "NumAlternates", "NumExpansions", "NumImplementations",
                       "IsReimplementation", "Kickstarted", "Rank:boardgame", "Rank:strategygames", "Rank:abstracts",
                       "Rank:familygames", "Rank:thematic", "Rank:cgs", "Rank:wargames", "Rank:partygames",
                       "Rank:childrensgames"]
    info_cols = [col for col in df.columns if col in info_candidates]

    used_cols = set(text_cols + summary_cols + info_cols + ["BGGId"])

    # Group 4: Binary features – assumed to be contiguous and coming next.
    try:
        low_exp_designer_idx = df.columns.get_loc("Low-Exp Designer")
    except KeyError:
        low_exp_designer_idx = None
    if low_exp_designer_idx is not None:
        binary_cols = [col for col in df.columns if
                       col not in used_cols and df.columns.get_loc(col) < low_exp_designer_idx]
    else:
        binary_cols = []

    # Group 5: Designers – these columns end with "Low-Exp Designer"
    if low_exp_designer_idx is not None:
        if binary_cols:
            start_designers_idx = df.columns.get_loc(binary_cols[-1]) + 1
        else:
            last_info_idx = max([df.columns.get_loc(col) for col in info_cols]) if info_cols else -1
            start_designers_idx = last_info_idx + 1
        designers_cols = list(df.columns[start_designers_idx: low_exp_designer_idx + 1])
    else:
        designers_cols = []

    # Group 6: Publishers – these columns end with "Low-Exp Publisher"
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
            if cols:  # only if group has columns
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
# Multi-Branch Game Encoder using ModuleDict for efficiency
##############################################
class MultiBranchGameEncoder(nn.Module):
    def __init__(self, groups_dims, fusion_out_dim=256,
                 branch_configs=None, designer_out_dim=32, publisher_out_dim=32):
        """
        groups_dims: dict mapping group names to their input dimensions.
        branch_configs: dict mapping group names to (hidden_layers, final_output_dim)
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
            x = features_dict["designers"]  # (B, num_designers)
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
# Rating Predictor Model
##############################################
class RatingPredictor(nn.Module):
    def __init__(self, game_encoder, num_users, num_games, latent_dim=256,
                 user_emb_dim=256, mlp_hidden_dim=1024, dropout_rate=0.5):
        super(RatingPredictor, self).__init__()
        self.game_encoder = game_encoder
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
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
            nn.Linear(mlp_hidden_dim // 4, 1)
        )

    def forward(self, game_features_dict, user_ids, game_ids):
        game_emb = self.game_encoder(game_features_dict)
        user_emb = self.user_embedding(user_ids)
        bias = self.game_bias(game_ids).squeeze(1)
        mlp_input = torch.cat([game_emb, user_emb], dim=1)
        rating_pred = self.mlp(mlp_input).squeeze(1)
        return rating_pred + bias + self.global_bias


##############################################
# Custom Loss Function: Inverse Frequency Weighted MSE
##############################################
class InverseFrequencyMSELoss(nn.Module):
    def __init__(self, rating_to_weight, min_rating=0.0, max_rating=10.0, precision=2):
        super(InverseFrequencyMSELoss, self).__init__()
        self.rating_to_weight = rating_to_weight
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.precision = precision
        scale = 10 ** precision
        num_bins = int(round((max_rating - min_rating) * scale)) + 1
        weight_list = []
        for i in range(num_bins):
            rating_val = min_rating + i / scale
            weight = rating_to_weight.get(round(rating_val, precision), 1.0)
            weight_list.append(weight)
        self.register_buffer('weight_tensor', torch.tensor(weight_list, dtype=torch.float32))
        self.scale = scale

    def forward(self, predictions, targets):
        indices = torch.round((targets - self.min_rating) * self.scale).long()
        indices = torch.clamp(indices, 0, self.weight_tensor.numel() - 1)
        indices = indices.to(self.weight_tensor.device)
        weights = self.weight_tensor[indices]
        loss = weights * (predictions - targets) ** 2
        return torch.mean(loss)


##############################################
# Unified training/evaluation function
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


def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, device='cpu'):
    train_losses = []
    test_losses = []
    min_loss = np.inf
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        tr_loss = process_epoch(model, train_loader, optimizer, criterion, device, train=True)
        te_loss = process_epoch(model, test_loader, optimizer, criterion, device, train=False)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {tr_loss:.4f}, Test Loss: {te_loss:.4f}")
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        if te_loss < min_loss:
            min_loss = te_loss
            torch.save(model.state_dict(), "ncf_model.pth")
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
            true_ratings.extend(ratings.cpu().numpy())
            predicted_ratings.extend(outputs.cpu().numpy())
    true_ratings = np.array(true_ratings) * sigma + mu
    predicted_ratings = np.array(predicted_ratings) * sigma + mu
    return true_ratings, predicted_ratings


##############################################
# Plotting functions
##############################################
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


##############################################
# Collate function for DataLoader
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
    ratings_df = pd.read_csv("../../bgg_data/user_ratings.csv")
    ratings_df = ratings_df[ratings_df["BGGId"].isin(set(games_df["BGGId"]))]
    unique_users = ratings_df['Username'].unique()
    unique_users = unique_users[1000:1500]
    ratings_df = ratings_df[ratings_df["Username"].isin(unique_users)]
    num_users = len(unique_users)
    user_id_map = {uid: idx + 1000 for idx, uid in enumerate(unique_users)}

    print(f"Total ratings: {len(ratings_df)}")
    train_ratings_df, test_ratings_df = preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42)
    print("Train-test split done.")

    mu_rating = train_ratings_df['Rating'].mean()
    sigma_rating = train_ratings_df['Rating'].std()
    print(f"Rating mean: {mu_rating:.2f}, std: {sigma_rating:.2f}")

    train_ratings_df['NormalizedRating'] = ((train_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)
    test_ratings_df['NormalizedRating'] = ((test_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)

    rating_to_weight = compute_inverse_frequency_weights_normalized(train_ratings_df)

    # Group and precompute game features.
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
    state_dict = torch.load("ncf_model.pth")
    model.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    criterion = InverseFrequencyMSELoss(rating_to_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 15
    print("Starting training...")
    train_losses, test_losses = train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion,
                                            device=device)

    plot_losses(train_losses, test_losses)

    true_test, pred_test = get_true_and_pred(model, test_loader, device, mu_rating, sigma_rating)
    plot_true_vs_pred(true_test, pred_test, title="Test Data: True vs Predicted Ratings")

    true_train, pred_train = get_true_and_pred(model, train_loader, device, mu_rating, sigma_rating)
    plot_true_vs_pred(true_train, pred_train, title="Train Data: True vs Predicted Ratings")


if __name__ == "__main__":
    main()
