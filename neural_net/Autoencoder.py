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
import torch.nn.functional as F
import math



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
        rating = row['Rating']

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
        # If user only has 2 ratings, this forces one to each split.
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


class GameDataset(Dataset):
    def __init__(self, games_df):
        self.games_df = games_df
        # Get features and handle NaN values first
        features = games_df.drop(columns=["BGGId", "Name"], errors='ignore').values
        
        # Replace NaN values with 0 before computing statistics
        features = np.nan_to_num(features, nan=0.0)
        
        # Compute statistics on clean data
        self.feature_means = np.nanmean(features, axis=0)
        self.feature_stds = np.nanstd(features, axis=0)
        
        # Add small epsilon to avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1e-6
        
        # Normalize features
        self.features = (features - self.feature_means) / self.feature_stds
        
        # Final safety check - replace any remaining NaNs or infs
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.games_df)

    def __getitem__(self, idx):
        features = self.features[idx]
        return torch.tensor(features, dtype=torch.float32)



class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder network with batch normalization and dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Decoder network with batch normalization and dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, input_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
        
    def train_model(self, train_loader, num_epochs=50, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for features in train_loader:
                features = features.to(self.device)
                     
                optimizer.zero_grad()
                decoded, encoded = self.forward(features)
                
                # Reconstruction loss
                recon_loss = criterion(decoded, features)
                
                # Stronger diversity loss
                batch_size = encoded.size(0)
                if batch_size > 1:
                    # Normalize embeddings
                    normalized_encoded = F.normalize(encoded, p=2, dim=1)
                    # Compute similarity matrix
                    similarity_matrix = torch.mm(normalized_encoded, normalized_encoded.t())
                    # Remove diagonal (self-similarity)
                    mask = torch.eye(batch_size, device=self.device) == 0
                    similarities = similarity_matrix[mask]
                    # Target similarity of 0.3 (we want more diverse embeddings)
                    diversity_loss = torch.mean((similarities - 0.3) ** 2)
                else:
                    diversity_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss with stronger diversity weight
                loss = recon_loss + 0.5 * diversity_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}')
        
        return train_losses
    
    def get_game_embeddings(self, game_features):
        """Get the latent representation (embedding) for a set of games."""
        self.eval()
        with torch.no_grad():
            game_features = game_features.to(self.device)
            _, encoded = self.forward(game_features)
        return encoded

    def recommend_games(self, user_games, all_games, top_k=10):
        """Recommend games based on similarity to user's favorite games."""
        # Get embeddings for user's games
        user_game_embeddings = self.get_game_embeddings(user_games)
        # Get embeddings for all games
        all_game_embeddings = self.get_game_embeddings(all_games)
        
        # Compute average embedding of user's games
        user_embedding = user_game_embeddings.mean(dim=0, keepdim=True)
        
        # Compute cosine similarity between user embedding and all games
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        all_game_embeddings = F.normalize(all_game_embeddings, p=2, dim=1)
        similarities = torch.mm(user_embedding, all_game_embeddings.t())
        
        # Get top-k most similar games
        top_scores, top_indices = torch.topk(similarities.squeeze(), k=top_k)
        
        return top_indices, top_scores


