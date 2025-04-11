# models/recommender_cbf.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from setup import *

def load_ratings(csv_path):
    """
    Loads and formats ratings CSV.
    """
    df = pd.read_csv(csv_path)
    return df[df["Rating"] > 0]  # Remove missing/unrated

def train_test_rating_split(ratings_df, test_size=0.2, random_state=42):
    """
    Returns train and test sets for user-item ratings.
    """
    return train_test_split(ratings_df, test_size=test_size, random_state=random_state)

def predict_ratings(test_df, train_df, sim_df, k=10):
    """
    Optimized function to predict test ratings based on similarity to items a user has rated in the train set.
    This version converts BGGIds to ints to ensure consistent comparisons.

    Args:
        test_df (pd.DataFrame): Test set with columns 'Username', 'BGGId', 'Rating'
        train_df (pd.DataFrame): Training set with columns 'Username', 'BGGId', 'Rating'
        sim_df (pd.DataFrame): Similarity matrix indexed and columned by BGGId (expected to be convertible to int)
        k (int): Number of most similar rated games to consider

    Returns:
        pd.DataFrame: Copy of test_df with an added column "Predicted" for the predicted ratings.
    """
    # Convert BGGId fields in test and train DataFrames to ints.
    test_df = test_df.copy()
    test_df['BGGId'] = test_df['BGGId'].astype(float).astype(int)
    train_df = train_df.copy()
    train_df['BGGId'] = train_df['BGGId'].astype(float).astype(int)

    # Convert similarity matrix indices and columns to ints.
    sim_df = sim_df.copy()
    sim_df.index = sim_df.index.astype(float).astype(int)
    sim_df.columns = sim_df.columns.astype(float).astype(int)

    # Build a mapping from game_id (int) to its index in the similarity matrix.
    game_to_idx = {game: i for i, game in enumerate(sim_df.index)}
    sim_matrix = sim_df.values  # NumPy array of shape (n_games, n_games).

    # Pre-group the training ratings by user.
    user_ratings_dict = {}
    for user, group in train_df.groupby('Username'):
        rated_ids = group['BGGId'].values  # ints now.
        ratings = group['Rating'].values.astype(float)
        valid_mask = np.array([gid in game_to_idx for gid in rated_ids])
        if np.any(valid_mask):
            user_ratings_dict[user] = (rated_ids[valid_mask], ratings[valid_mask])
        else:
            user_ratings_dict[user] = (np.array([]), np.array([]))

    predicted_ratings = []
    for _, row in test_df.iterrows():
        user = row['Username']
        target_game = int(row['BGGId'])

        # Skip if the target game is not present in similarity matrix.
        if target_game not in game_to_idx:
            predicted_ratings.append(np.nan)
            continue

        target_idx = game_to_idx[target_game]

        # Retrieve user's rated games from training set.
        if user not in user_ratings_dict or len(user_ratings_dict[user][0]) == 0:
            predicted_ratings.append(np.nan)
            continue

        rated_ids, ratings = user_ratings_dict[user]
        rated_indices = np.array([game_to_idx[gid] for gid in rated_ids])
        sims = sim_matrix[target_idx, rated_indices]

        # If more than k items are available, select the top k similar ones.
        if len(sims) > k:
            top_k_indices = np.argsort(sims)[-k:]
            sims = sims[top_k_indices]
            ratings = ratings[top_k_indices]

        # Compute the weighted average if the sum of similarities is positive.
        if np.sum(sims) > 0:
            prediction = np.dot(sims, ratings) / np.sum(sims)
        else:
            prediction = np.nan

        predicted_ratings.append(prediction)

    test_df["Predicted"] = predicted_ratings
    return test_df



