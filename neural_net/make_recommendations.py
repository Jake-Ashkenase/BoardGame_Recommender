import torch
import pandas as pd
from Autoencoder import Autoencoder
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def load_model_and_data(model_path='autoencoder_model.pth'):
    # Load games data
    games_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bgg_data_all/overall_games.csv"))
    games_df = pd.read_csv(games_csv_path)
    
    # Initialize model
    input_dim = games_df.drop(columns=["BGGId", "Name"], errors='ignore').shape[1]
    hidden_dim = 256
    model = Autoencoder(input_dim, hidden_dim)
    
    # Try to load the model
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    except RuntimeError as e:
        print("Error loading model. Architecture mismatch detected.")
        print("Please retrain the model with the current architecture.")
        raise e
    
    model.eval()
    
    return model, games_df

def get_game_features(games_df, game_ids):
    """Get feature tensors for a list of game IDs"""
    features = []
    for game_id in game_ids:
        game_features = games_df[games_df["BGGId"] == game_id].drop(columns=["BGGId", "Name"], errors='ignore')
        if not game_features.empty:
            features.append(game_features.iloc[0].values)
        else:
            print(f"Warning: Game ID {game_id} not found in database")
    return torch.tensor(features, dtype=torch.float32)

def get_recommendations(user_game_ids, model, games_df, top_k=10):
    """
    Get game recommendations based on user's favorite games.
    
    Args:
        user_game_ids: List of BGGIds for user's favorite games
        model: Trained autoencoder model
        games_df: DataFrame containing all games
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame containing recommended games with their similarity scores
    """
    # Get features for user's games
    user_games = get_game_features(games_df, user_game_ids)
    
    # Get features for all games
    all_games = get_game_features(games_df, games_df["BGGId"].tolist())
    
    # Get recommendations
    recommended_indices, scores = model.recommend_games(user_games, all_games, top_k=top_k)
    
    # Get recommended games
    recommended_games = games_df.iloc[recommended_indices].copy()
    
    # Add scores
    recommended_games["Similarity_Score"] = scores.cpu().numpy()
    
    return recommended_games[["BGGId", "Name", "Similarity_Score"]]

def visualize_embeddings(model, games_df, num_games=1000):
    """
    Visualize game embeddings using TensorBoard.
    
    Args:
        model: Trained autoencoder model
        games_df: DataFrame containing all games
        num_games: Number of games to visualize (for performance)
    """
    # Clean up any existing runs
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    
    # Get a subset of games
    games_subset = games_df.sample(n=min(num_games, len(games_df)), random_state=42)
    
    # Get features for the subset
    game_features = get_game_features(games_df, games_subset["BGGId"].tolist())
    
    # Get embeddings
    embeddings = model.get_game_embeddings(game_features)
    embeddings = embeddings.cpu().numpy()
    
    # Create metadata with more information
    metadata = []
    for _, row in games_subset.iterrows():
        metadata.append([row['Name'], str(row['BGGId'])])
    
    # Create TensorBoard writer
    writer = SummaryWriter('runs/game_embeddings')
    
    # Add embeddings to TensorBoard with metadata
    writer.add_embedding(
        embeddings,
        metadata=metadata,
        metadata_header=['Name', 'BGGId'],
        tag='game_embeddings',
        global_step=0
    )
    
    # Add some additional information
    writer.add_text('embedding_info', 
                   f'Total games visualized: {len(games_subset)}\n'
                   f'Embedding dimension: {embeddings.shape[1]}')
    
    writer.close()
    print("Embeddings have been added to TensorBoard. Run the following command to view:")
    print("tensorboard --logdir=runs")

def evaluate_reconstruction(model, games_df, num_games=1000):
    """
    Evaluate the autoencoder's reconstruction performance.
    
    Args:
        model: Trained autoencoder model
        games_df: DataFrame containing all games
        num_games: Number of games to evaluate
    """
    # Get a subset of games
    games_subset = games_df.sample(n=min(num_games, len(games_df)), random_state=42)
    
    # Get features for the subset
    game_features = get_game_features(games_df, games_subset["BGGId"].tolist())
    
    # Get reconstructions
    model.eval()
    with torch.no_grad():
        reconstructed, _ = model(game_features)
    
    # Convert to numpy for evaluation
    original = game_features.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)
    
    # Calculate reconstruction error per feature
    feature_errors = np.mean(np.abs(original - reconstructed), axis=0)
    
    print("\nReconstruction Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Plot feature-wise reconstruction errors
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_errors)), feature_errors)
    plt.title('Reconstruction Error per Feature')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('feature_reconstruction_errors.png')
    plt.close()
    
    return mse, mae

def evaluate_embedding_quality(model, games_df, num_games=1000):
    """
    Evaluate the quality of the learned embeddings.
    
    Args:
        model: Trained autoencoder model
        games_df: DataFrame containing all games
        num_games: Number of games to evaluate
    """
    # Get a subset of games
    games_subset = games_df.sample(n=min(num_games, len(games_df)), random_state=42)
    
    # Get features and embeddings
    game_features = get_game_features(games_df, games_subset["BGGId"].tolist())
    embeddings = model.get_game_embeddings(game_features)
    embeddings = embeddings.cpu().numpy()
    
    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    # Plot similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50)
    plt.title('Distribution of Embedding Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.savefig('embedding_similarities.png')
    plt.close()
    
    # Print statistics
    print("\nEmbedding Quality Metrics:")
    print(f"Mean similarity: {np.mean(similarities):.4f}")
    print(f"Similarity std: {np.std(similarities):.4f}")
    print(f"Min similarity: {np.min(similarities):.4f}")
    print(f"Max similarity: {np.max(similarities):.4f}")
    
    return np.mean(similarities), np.std(similarities)

def main():
    # Example usage
    model, games_df = load_model_and_data()
    
    # Evaluate reconstruction performance
    mse, mae = evaluate_reconstruction(model, games_df)
    
    # Evaluate embedding quality
    mean_sim, sim_std = evaluate_embedding_quality(model, games_df)
    
    # Visualize embeddings
    visualize_embeddings(model, games_df)
    
    # Example: User's 5 favorite games (replace with actual BGGIds)
    user_game_ids = [181.0, 320.0, 5244.0, 5260.0, 5423.0]
    
    # Get recommendations
    recommendations = get_recommendations(user_game_ids, model, games_df)
    
    # Print recommendations
    print("\nRecommended Games:")
    print(recommendations)

if __name__ == "__main__":
    main() 