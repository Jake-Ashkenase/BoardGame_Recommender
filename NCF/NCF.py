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


# def silent_tensor_repr(self):
#     return ""
# torch.Tensor.__repr__ = silent_tensor_repr


class RatingsDataset(Dataset):
    def __init__(self, ratings_df, games_df, user_id_map, transform=None):
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.games_df = games_df
        self.user_id_map = user_id_map
        self.transform = transform
        if torch.utils.data.get_worker_info() is not None:
            sys.stdout = open(os.devnull, 'w')

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_id = row['Username']
        bggid = row['BGGId']
        rating = row['Rating']

        # Map user to index
        user_index = self.user_id_map[user_id]

        # Safely get game features
        game_features = self.games_df[self.games_df["BGGId"] == bggid]
        if not game_features.empty:
            game_features = game_features.iloc[0].astype(np.float32).values  # 1D numpy array
        else:
            # Handle missing game gracefully (e.g., return zeros)
            game_features = np.zeros(self.games_df.shape[1], dtype=np.float32)

        game_features = torch.tensor(game_features, dtype=torch.float32)
        user_index = torch.tensor(user_index, dtype=torch.long)
        rating = torch.tensor(rating, dtype=torch.float32)

        return game_features, user_index, rating


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


class RatingPredictorOld(nn.Module):
    def __init__(self, game_encoder, num_users, latent_dim=64, user_emb_dim=32, mlp_hidden_dim=128):
        super(RatingPredictorOld, self).__init__()
        self.game_encoder = game_encoder
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + user_emb_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

    def forward(self, game_features, user_ids):
        game_emb = self.game_encoder(game_features)
        user_emb = self.user_embedding(user_ids)
        x = torch.cat([game_emb, user_emb], dim=1)
        rating = self.mlp(x)
        return rating.squeeze(1)


class RatingPredictor(nn.Module):
    def __init__(self, game_encoder, num_users, num_games, latent_dim=64, user_emb_dim=32, mlp_hidden_dim=128):
        super(RatingPredictor, self).__init__()
        self.game_encoder = game_encoder
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        # Create an embedding for game bias
        self.game_bias = nn.Embedding(num_games, 1)
        # Global bias as a parameter
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + user_emb_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )

    def forward(self, game_features, user_ids, game_ids):
        game_emb = self.game_encoder(game_features)
        user_emb = self.user_embedding(user_ids)
        # Get biases for game and add global bias
        game_bias = self.game_bias(game_ids).squeeze(1)
        # Compute rating from MLP
        mlp_input = torch.cat([game_emb, user_emb], dim=1)
        rating_pred = self.mlp(mlp_input).squeeze(1)
        # Add bias terms
        rating_pred = rating_pred + game_bias + self.global_bias
        return rating_pred


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for game_features, user_ids, ratings in loader:
        game_features = game_features.to(device)
        user_ids = user_ids.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad()
        outputs = model(game_features, user_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * game_features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for game_features, user_ids, ratings in loader:
            game_features = game_features.to(device)
            user_ids = user_ids.to(device)
            ratings = ratings.to(device)
            outputs = model(game_features, user_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item() * game_features.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, device='cpu'):
    train_losses = []
    test_losses = []
    #for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
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


def get_true_and_pred(model, loader, device):
    model.eval()
    true_ratings = []
    predicted_ratings = []
    with torch.no_grad():
        for game_features, user_ids, ratings in loader:
            game_features = game_features.to(device)
            user_ids = user_ids.to(device)
            ratings = ratings.to(device)
            outputs = model(game_features, user_ids)
            true_ratings.extend(ratings.cpu().numpy())
            predicted_ratings.extend(outputs.cpu().numpy())

    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    return true_ratings, predicted_ratings

def plot_true_vs_pred(true_ratings, predicted_ratings, title="True Ratings vs. Predicted Ratings"):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_ratings, predicted_ratings, alpha=0.5, label="Predictions")

    # Create a y=x line for perfect predictions.
    min_val = min(true_ratings.min(), predicted_ratings.min())
    max_val = max(true_ratings.max(), predicted_ratings.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

    plt.xlabel("True Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # get and format game data
    games_df = pd.read_csv("../../bgg_data/overall_games.csv")
    names = games_df[["Name", "BGGId"]]
    games_df = games_df.drop(columns=["Name"], errors='ignore')

    # get and format user ratings data
    ratings_df = pd.read_csv("../../bgg_data/user_ratings.csv")
    ratings_df = ratings_df[ratings_df["BGGId"].isin(set(games_df["BGGId"]))]
    # train with just the first 3k users (bc 411k is taking wayyyyy too long)
    unique_users = ratings_df['Username'].unique()[500:1000]
    ratings_df = ratings_df[ratings_df["Username"].isin(unique_users)]
    user_id_map = {uid: idx + 500 for idx, uid in enumerate(unique_users)}
    print(len(ratings_df))
    train_ratings_df, test_ratings_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print("train test split done")

    train_dataset = RatingsDataset(train_ratings_df, games_df, user_id_map)
    test_dataset = RatingsDataset(test_ratings_df, games_df, user_id_map)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=silence_worker_output)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=silence_worker_output)
    print("data loaders created")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    latent_dim = 64
    input_dim = len(train_dataset.games_df.columns)
    num_users = len(unique_users)
    game_encoder = GameEncoder(input_dim, latent_dim=latent_dim)
    model = RatingPredictor(game_encoder, num_users, latent_dim=latent_dim, user_emb_dim=32, mlp_hidden_dim=128)
    state_dict = torch.load("ncf_model_v4.pth")
    model.load_state_dict(state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.0005)
    num_epochs = 30
    print("model created")

    train_loss, test_loss = train_model(
        model,
        num_epochs,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device=device)
    # del torch.Tensor.__repr__  # restores default behavior


    plot_losses(train_loss, test_loss)

    true, pred = get_true_and_pred(model, test_loader, device)
    plot_true_vs_pred(true, pred)

    torch.save(model.state_dict(), "ncf_model_v4.pth")


if __name__ == "__main__":
    main()
