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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SetTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_blocks, d_ff, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.blocks = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_blocks)
        ])
        
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Pool across sequence dimension (mean pooling)
        # this is the biggest different between transformer and set transformer
        # we are aggreagating information across all set members in a permutation-invariant way 
        x = x.mean(dim=1)
        
        # Project to rating prediction
        x = self.output_projection(x)
        
        return x

class SetTransformerRecommender(nn.Module):
    def __init__(self, game_feature_dim, num_users, d_model=256, num_heads=8, num_blocks=4, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # User embeddings
        self.user_embeddings = nn.Embedding(num_users, d_model)
        
        # Game feature processing
        self.game_encoder = nn.Sequential(
            nn.Linear(game_feature_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Set Transformer for processing game features
        self.set_transformer = SetTransformer(
            input_dim=d_model,
            d_model=d_model,
            num_heads=num_heads,
            num_blocks=num_blocks,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Final prediction layers - adjusted dimensions
        self.final_layers = nn.Sequential(
            nn.Linear(d_model, d_model),  # Process combined features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # Output single rating
        )
        
    def forward(self, game_features, user_indices):
        # If game features are 2D, add a batch dimension
        if len(game_features.shape) == 2:
            game_features = game_features.unsqueeze(1)

        # Process game features
        game_encoded = self.game_encoder(game_features)
        
        # Get user embeddings
        user_embeddings = self.user_embeddings(user_indices)

        # Process game features through Set Transformer
        game_transformed = self.set_transformer(game_encoded)
        
        # Combine user embeddings and transformed game features
        # Instead of concatenating, we add them to maintain the same dimension
        combined = user_embeddings + game_transformed
        
        # Final prediction
        rating = self.final_layers(combined)
        
        return rating.squeeze(-1)

    def recommend_games(self, user_id, user_rated_games, all_games, top_k=10):
        """
        Recommend games for a user based on their previous ratings.
        
        Args:
            user_id: The ID of the user to recommend for
            user_rated_games: Tensor of game features for games the user has rated
            all_games: Tensor of game features for all available games
            top_k: Number of recommendations to return
            
        Returns:
            Tuple of (recommended_game_indices, similarity_scores)
        """
        # Process user's rated games through Set Transformer
        if len(user_rated_games.shape) == 2:
            user_rated_games = user_rated_games.unsqueeze(0)
        
        # Encode rated games
        rated_games_encoded = self.game_encoder(user_rated_games)
        user_preferences = self.set_transformer(rated_games_encoded)
        
        # Process all available games
        if len(all_games.shape) == 2:
            all_games = all_games.unsqueeze(0)
        all_games_encoded = self.game_encoder(all_games)
        all_games_transformed = self.set_transformer(all_games_encoded)
        
        # Get user embedding
        user_embedding = self.user_embeddings(torch.tensor([user_id], device=user_rated_games.device))
        
        # Compute similarity scores
        # For each game, combine with user embedding and get score
        scores = []
        for game_features in all_games_transformed.squeeze(0):
            # Add user embedding to game features instead of concatenating
            combined = user_embedding + game_features.unsqueeze(0)
            score = self.final_layers(combined)
            scores.append(score.item())
        
        # Convert to tensor and get top-k
        scores = torch.tensor(scores)
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        return top_indices, top_scores

def train_recommendation_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.0005, device='cuda'):
    """
    Train the model with a focus on learning good game representations for recommendations.
    Uses a combination of rating prediction and contrastive learning.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            game_features, user_indices, ratings, game_indices = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Get predictions
            predictions = model(game_features, user_indices)
            
            # Rating prediction loss
            rating_loss = criterion(predictions, ratings)
            
            # Contrastive loss to learn good game representations
            # Get game embeddings
            game_encoded = model.game_encoder(game_features)
            game_transformed = model.set_transformer(game_encoded)
            
            # Normalize embeddings
            game_transformed = F.normalize(game_transformed, p=2, dim=-1)
            
            # Compute similarity matrix
            similarity_matrix = torch.mm(game_transformed, game_transformed.t())
            
            # Create positive and negative pairs
            batch_size = game_transformed.size(0)
            labels = torch.arange(batch_size, device=device)
            
            # Contrastive loss
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
            
            # Combined loss
            loss = rating_loss + 0.1 * contrastive_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                game_features, user_indices, ratings, game_indices = [b.to(device) for b in batch]
                predictions = model(game_features, user_indices)
                loss = criterion(predictions, ratings)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}')
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'set_transformer_recommender.pth')
    
    return train_losses, test_losses


def main():
    # Get and format game data.
    games_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bgg_data_all/overall_games.csv"))
    games_df = pd.read_csv(games_csv_path)
    names = games_df[["Name", "BGGId"]]
    games_df = games_df.drop(columns=["Name"], errors='ignore')

    # Create a mapping from BGGId to a game index.
    unique_game_ids = games_df["BGGId"].unique()
    game_id_map = {bgid: idx for idx, bgid in enumerate(unique_game_ids)}

    # Get and format user ratings data.
    ratings_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bgg_data_all/user_ratings.csv"))
    ratings_df = pd.read_csv(ratings_csv_path)
    ratings_df = ratings_df[ratings_df["BGGId"].isin(set(games_df["BGGId"]))]

    # Train with a subset of users.
    unique_users = ratings_df['Username'].unique()
    num_users = len(unique_users)
    unique_users = unique_users[2000:2100]
    ratings_df = ratings_df[ratings_df["Username"].isin(unique_users)]
    # Create user mapping without offset
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    print('Number of Ratings: ', len(ratings_df))
    train_ratings_df, test_ratings_df = preprocess_train_test_split(ratings_df, test_size=0.2, random_state=42)
    print("train test split done")

    train_dataset = RatingsDataset(train_ratings_df, games_df, user_id_map, game_id_map)
    test_dataset = RatingsDataset(test_ratings_df, games_df, user_id_map, game_id_map)

    # Create data loaders (using multiple workers causes issues for some reason)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Initialize the Set Transformer model
    game_feature_dim = games_df.drop(columns=["BGGId"], errors='ignore').shape[1]
    num_users = len(user_id_map)
    
    model = SetTransformerRecommender(
        game_feature_dim=game_feature_dim,
        num_users=num_users,
        d_model=256,
        num_heads=8,
        num_blocks=4,
        d_ff=1024,
        dropout=0.1
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Train the model
    train_losses, test_losses = train_recommendation_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        learning_rate=0.0005,
        device=device
    )

    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.savefig('set_transformer_training.png')
    plt.close()

if __name__ == "__main__":
    main()