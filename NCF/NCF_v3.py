import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def silence_worker_output(worker_id):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Updated RatingsDataset that returns game_ids and uses normalized ratings if available.
class RatingsDataset(Dataset):
    def __init__(self, ratings_df, games_df, user_id_map, game_id_map, transform=None):
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.games_df = games_df
        self.user_id_map = user_id_map
        self.game_id_map = game_id_map
        self.transform = transform
        if torch.utils.data.get_worker_info() is not None:
            sys.stdout = open(os.devnull, 'w')

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_id = row['Username']
        bggid = row['BGGId']
        # Use normalized rating if available, otherwise fall back to original.
        rating = row['NormalizedRating'] if 'NormalizedRating' in row else row['Rating']

        # Map user to index.
        user_index = self.user_id_map[user_id]

        # Safely get game features.
        game_features_df = self.games_df[self.games_df["BGGId"] == bggid]
        if not game_features_df.empty:
            # Remove BGGId column so only numeric features remain.
            game_features = game_features_df.drop(columns=["BGGId"], errors='ignore').iloc[0].astype(np.float32).values
        else:
            # Handle missing game gracefully (return zeros).
            feature_dim = self.games_df.drop(columns=["BGGId"], errors='ignore').shape[1]
            game_features = np.zeros(feature_dim, dtype=np.float32)

        # Convert to tensors.
        game_features = torch.tensor(game_features, dtype=torch.float32)
        user_index = torch.tensor(user_index, dtype=torch.long)
        rating = torch.tensor(rating, dtype=torch.float32)

        # Look up the game index from game_id_map.
        game_index = self.game_id_map[bggid]
        game_index = torch.tensor(game_index, dtype=torch.long)

        return game_features, user_index, rating, game_index


# GameEncoder remains unchanged.
class GameEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(GameEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# Updated RatingPredictor that expects game_ids and uses a game bias embedding.
class RatingPredictor(nn.Module):
    def __init__(self, game_encoder, num_users, num_games, latent_dim=256, user_emb_dim=256, mlp_hidden_dim=2048,
                 dropout_rate=0.5):
        super(RatingPredictor, self).__init__()
        self.game_encoder = game_encoder
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        # Embedding for game bias.
        self.game_bias = nn.Embedding(num_games, 1)
        # Global bias parameter.
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Input size for the MLP is the concatenated game and user embeddings.
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

    def forward(self, game_features, user_ids, game_ids):
        game_emb = self.game_encoder(game_features)
        user_emb = self.user_embedding(user_ids)
        # Look up game bias.
        bias = self.game_bias(game_ids).squeeze(1)
        mlp_input = torch.cat([game_emb, user_emb], dim=1)
        rating_pred = self.mlp(mlp_input).squeeze(1)
        # Add the bias terms.
        rating_pred = rating_pred + bias + self.global_bias
        return rating_pred


# Custom Loss Function: Inverse Rating Frequency Weighted MSE.
# Here the targets are assumed to be the normalized ratings.
class InverseFrequencyMSELoss(nn.Module):
    def __init__(self, rating_to_weight, min_rating=0.0, max_rating=10.0, precision=2, device="mps"):
        """
        rating_to_weight: dict mapping normalized rating (rounded to `precision` decimals) to its inverse frequency weight.
        min_rating, max_rating: the range of possible ratings.
        precision: number of decimal places to round.
        """
        super(InverseFrequencyMSELoss, self).__init__()
        self.rating_to_weight = rating_to_weight
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.precision = precision

        # Compute number of bins. For 2 decimals, scale factor is 100.
        scale = 10 ** precision
        num_bins = int(round((max_rating - min_rating) * scale)) + 1

        # Build the weight tensor by iterating over all possible rounded rating values.
        weight_list = []
        for i in range(num_bins):
            rating_val = min_rating + i / scale
            # Use the weight from the dictionary, defaulting to 1.0 if not found.
            weight = rating_to_weight.get(round(rating_val, precision), 1.0)
            weight_list.append(weight)
        # Register the weight tensor as a buffer so it's moved to the right device.
        self.register_buffer('weight_tensor', torch.tensor(weight_list, dtype=torch.float32))
        self.scale = scale

    def forward(self, predictions, targets):
        # Compute indices for each target.
        indices = torch.round((targets - self.min_rating) * self.scale).long()
        indices = torch.clamp(indices, 0, self.weight_tensor.numel() - 1)
        # Move indices to the same device as weight_tensor.
        indices = indices.to(self.weight_tensor.device)
        # Look up weights in a vectorized manner.
        weights = self.weight_tensor[indices]
        loss = weights * (predictions - targets) ** 2
        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for game_features, user_ids, ratings, game_ids in loader:
        game_features = game_features.to(device)
        user_ids = user_ids.to(device)
        ratings = ratings.to(device)
        game_ids = game_ids.to(device)
        optimizer.zero_grad()
        outputs = model(game_features, user_ids, game_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * game_features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for game_features, user_ids, ratings, game_ids in loader:
            game_features = game_features.to(device)
            user_ids = user_ids.to(device)
            ratings = ratings.to(device)
            game_ids = game_ids.to(device)
            outputs = model(game_features, user_ids, game_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item() * game_features.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, device='cpu'):
    train_losses = []
    test_losses = []
    min_loss = np.inf
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        min_loss = min(min_loss, test_loss)
        if test_loss == min_loss:
            path = "ncf_model_v8.pth"
            torch.save(model.state_dict(), path)
    return train_losses, test_losses


def plot_losses(train_losses, test_losses, title="Training and Test Loss Over Epochs"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_true_and_pred(model, loader, device, mu, sigma):
    """
    Returns predictions and true ratings in the original scale.
    The loader yields normalized ratings; we invert the normalization.
    """
    model.eval()
    true_ratings = []
    predicted_ratings = []
    with torch.no_grad():
        for game_features, user_ids, ratings, game_ids in loader:
            game_features = game_features.to(device)
            user_ids = user_ids.to(device)
            ratings = ratings.to(device)
            game_ids = game_ids.to(device)
            outputs = model(game_features, user_ids, game_ids)
            true_ratings.extend(ratings.cpu().numpy())
            predicted_ratings.extend(outputs.cpu().numpy())
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    # Invert normalization: original = normalized * sigma + mu.
    true_ratings = true_ratings * sigma + mu
    predicted_ratings = predicted_ratings * sigma + mu
    return true_ratings, predicted_ratings


def plot_true_vs_pred(true_ratings, predicted_ratings, title="True Ratings vs. Predicted Ratings"):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_ratings, predicted_ratings, alpha=0.5, label="Predictions")
    min_val = min(true_ratings.min(), predicted_ratings.min())
    max_val = max(true_ratings.max(), predicted_ratings.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    plt.xlabel("True Ratings (Original Scale)")
    plt.ylabel("Predicted Ratings (Original Scale)")
    plt.title(title)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(true, pred, save_as=None, aggregate=True, shaded_region=True, smoothing_window=5):
    plt.figure(figsize=(8, 6), facecolor='#384957')

    if aggregate or shaded_region:
        # Convert to Pandas DataFrame for efficient grouping.
        df = pd.DataFrame({"actual": true, "predicted": pred})
        grouped = df.groupby("actual")["predicted"]

        # Compute mean, min, and max for each actual rating value.
        mean_pred = grouped.mean()
        min_pred = grouped.min()
        max_pred = grouped.max()

        # Sort by actual rating for proper plotting.
        sorted_actual = mean_pred.index

        # Apply moving average smoothing.
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

    # Plot the perfect prediction line.
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--', label="Perfect Prediction", color="#384957")
    plt.xlabel("Actual", color="white")
    plt.ylabel("Predicted", color="white")
    plt.title("Predicted vs. Actual", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
    plt.show()



def preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42):
    # Remove users with fewer than 2 ratings.
    user_counts = ratings_df['Username'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    ratings_df = ratings_df[ratings_df['Username'].isin(valid_users)]

    # Remove games with fewer than 2 ratings.
    game_counts = ratings_df['BGGId'].value_counts()
    valid_games = game_counts[game_counts >= 2].index
    ratings_df = ratings_df[ratings_df['BGGId'].isin(valid_games)]

    # Split ratings per user so that each user has at least one rating in test and train.
    train_list = []
    test_list = []
    rng = np.random.RandomState(random_state)
    for user, group in ratings_df.groupby('Username'):
        group = group.sample(frac=1, random_state=rng.randint(0, 10000))  # shuffle
        num_ratings = len(group)
        # Ensure at least one rating in test (and one in train).
        num_test = max(1, int(np.floor(num_ratings * test_size)))
        test_ratings = group.iloc[:num_test]
        train_ratings = group.iloc[num_test:]
        if train_ratings.empty:
            # In case all ratings fall in test, force one into train.
            train_ratings = test_ratings.iloc[[0]]
            test_ratings = test_ratings.iloc[1:]
        train_list.append(train_ratings)
        test_list.append(test_ratings)

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Now, ensure every game appears in both sets.
    all_games = set(ratings_df['BGGId'].unique())
    games_in_train = set(train_df['BGGId'].unique())
    games_in_test = set(test_df['BGGId'].unique())

    missing_in_train = all_games - games_in_train
    missing_in_test = all_games - games_in_test

    # For games missing in train, move one rating from test to train.
    for game in missing_in_train:
        candidate_rows = test_df[test_df['BGGId'] == game]
        if not candidate_rows.empty:
            row_to_move = candidate_rows.iloc[[0]]
            train_df = pd.concat([train_df, row_to_move], ignore_index=True)
            test_df = test_df.drop(candidate_rows.index[0])

    # For games missing in test, move one rating from train to test.
    for game in missing_in_test:
        candidate_rows = train_df[train_df['BGGId'] == game]
        if not candidate_rows.empty:
            row_to_move = candidate_rows.iloc[[0]]
            test_df = pd.concat([test_df, row_to_move], ignore_index=True)
            train_df = train_df.drop(candidate_rows.index[0])

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_inverse_frequency_weights_normalized(ratings_df):
    """
    Assumes that ratings_df contains the 'NormalizedRating' column.
    Returns a dictionary mapping each normalized rating (rounded to 2 decimals) to 1/(frequency).
    """
    rating_counts = ratings_df['NormalizedRating'].value_counts().to_dict()
    rating_to_weight = {round(rating, 2): 1.0 / count for rating, count in rating_counts.items()}
    return rating_to_weight


def main():
    # Get and format game data.
    games_df = pd.read_csv("../../bgg_data/overall_games.csv")
    names = games_df[["Name", "BGGId"]]
    games_df = games_df.drop(columns=["Name"], errors='ignore')

    # Create a mapping from BGGId to a game index.
    unique_game_ids = games_df["BGGId"].unique()
    game_id_map = {bgid: idx for idx, bgid in enumerate(unique_game_ids)}

    # Get and format user ratings data.
    ratings_df = pd.read_csv("../../bgg_data/user_ratings.csv")
    ratings_df = ratings_df[ratings_df["BGGId"].isin(set(games_df["BGGId"]))]
    # Train with a subset of users.
    unique_users = ratings_df['Username'].unique()
    num_users = len(unique_users)
    unique_users = unique_users[3000:3500]
    ratings_df = ratings_df[ratings_df["Username"].isin(unique_users)]
    user_id_map = {uid: idx + 3000 for idx, uid in enumerate(unique_users)}
    print(len(ratings_df))
    train_ratings_df, test_ratings_df = preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42)
    print("train test split done")

    # Compute normalization parameters from the original train_eval ratings.
    mu_rating = train_ratings_df['Rating'].mean()
    sigma_rating = train_ratings_df['Rating'].std()
    print(f"Rating mean: {mu_rating:.2f}, std: {sigma_rating:.2f}")

    # Add a normalized ratings column (rounded to 2 decimals) for stability.
    train_ratings_df['NormalizedRating'] = ((train_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)
    test_ratings_df['NormalizedRating'] = ((test_ratings_df['Rating'] - mu_rating) / sigma_rating).round(2)

    # Precompute inverse frequency weights using the normalized ratings.
    rating_to_weight = compute_inverse_frequency_weights_normalized(train_ratings_df)
    #print("Rating weights (normalized):", rating_to_weight)

    train_dataset = RatingsDataset(train_ratings_df, games_df, user_id_map, game_id_map)
    test_dataset = RatingsDataset(test_ratings_df, games_df, user_id_map, game_id_map)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              worker_init_fn=silence_worker_output)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                             worker_init_fn=silence_worker_output)
    print("data loaders created")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    latent_dim = 256
    # Note: input_dim is the number of numeric columns (excluding BGGId) in games_df.
    input_dim = len(games_df.drop(columns=["BGGId"], errors='ignore').columns)
    num_games = len(unique_game_ids)
    game_encoder = GameEncoder(input_dim, latent_dim=latent_dim)
    model = RatingPredictor(game_encoder, num_users, num_games, latent_dim=latent_dim, user_emb_dim=256,
                            mlp_hidden_dim=1024)
    state_dict = torch.load("ncf_model_v8.pth")
    model.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    # Use the custom inverse frequency weighted MSE loss.
    criterion = InverseFrequencyMSELoss(rating_to_weight)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 10
    print("model created")

    train_loss, test_loss = train_model(
        model,
        num_epochs,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device=device)

    plot_losses(train_loss, test_loss)

    # Get predictions and invert normalization to the original rating scale.
    true, pred = get_true_and_pred(model, test_loader, device, mu_rating, sigma_rating)
    #plot_true_vs_pred(true, pred)
    plot_predictions(true, pred)
    true, pred = get_true_and_pred(model, train_loader, device, mu_rating, sigma_rating)
    # plot_true_vs_pred(true, pred)
    plot_predictions(true, pred)


if __name__ == "__main__":
    main()
