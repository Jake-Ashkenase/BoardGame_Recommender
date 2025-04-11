#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np

from models import load_ratings, train_test_rating_split, predict_ratings
from setup import RATINGS_CSV_PATH, GAMES_SIMILARITIES_CSV_PATH
from train_eval import plot_predictions


def main(ratings_csv, similarity_csv, plot_output):
    # Load the ratings data.
    ratings_df = load_ratings(ratings_csv)
    print(f"Loaded {len(ratings_df)} ratings from {ratings_csv}.")

    # Split into training and test sets.
    train_df, test_df = train_test_rating_split(ratings_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} ratings; Test set: {len(test_df)} ratings.")

    # Load the similarity matrix CSV.
    sim_df = pd.read_csv(similarity_csv, index_col=0)
    print(f"Similarity matrix loaded from {similarity_csv} with shape {sim_df.shape}.")

    # Predict ratings for the test set based on user's rated games in the train set.
    test_predictions_df = predict_ratings(test_df, train_df, sim_df, k=10)
    print("Predictions computed for test set.")

    # Extract true and predicted ratings for plotting.
    true_ratings = test_predictions_df["Rating"].values.tolist()
    predicted_ratings = test_predictions_df["Predicted"].values.tolist()
    # print(true_ratings.shape, predicted_ratings.shape)
    print(predicted_ratings[:10])
    # Create a True vs. Predicted plot.
    plot_predictions(true_ratings, predicted_ratings, save_as=plot_output, smoothing_window=20)
    print(f"True vs. Predicted plot saved to {plot_output}")

    # Optionally, save the test predictions with predictions to CSV.
    test_predictions_df.to_csv("test_predictions_with_predicted.csv", index=False)
    print("Test predictions saved to 'test_predictions_with_predicted.csv'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate content-based recommendations via latent embeddings and similarity."
    )
    parser.add_argument("--ratings_csv", type=str, default=RATINGS_CSV_PATH,
                        help="Path to the ratings CSV file.")
    parser.add_argument("--similarity_csv", type=str, default=GAMES_SIMILARITIES_CSV_PATH,
                        help="Path to the previously saved similarity matrix CSV file.")
    parser.add_argument("--plot_output", type=str, default="true_vs_predicted.png",
                        help="Filename to save the true vs. predicted plot.")

    args = parser.parse_args()
    main(args.ratings_csv, args.similarity_csv, args.plot_output)
