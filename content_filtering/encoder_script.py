#!/usr/bin/env python
import argparse
import pandas as pd
import torch
import numpy as np

from models import *
from setup import *


def encode_all_games(csv_path, output_csv_path, model_weights_path="../../bgg_data/denoising_autoencoder.pth"):
    # Load the games CSV.
    df = pd.read_csv(csv_path)

    # Preserve the game identifiers and names.
    bggids = df["BGGId"] if "BGGId" in df.columns else None
    names = df["Name"] if "Name" in df.columns else None

    # Drop BGGId and Name to obtain features for encoding.
    df_features = df.drop(columns=["BGGId", "Name"], errors='ignore')
    data = df_features.values.astype(np.float32)
    input_dim = data.shape[1]

    # Set up device and load the autoencoder.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder(input_dim).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()  # Ensure model is in evaluation mode.

    # Convert features into a torch tensor.
    data_tensor = torch.from_numpy(data).to(device)

    # Encode all games using the model's built-in encode method.
    with torch.no_grad():
        latent = model.encode(data_tensor)
    latent = latent.cpu().numpy()

    # Build column names for latent dimensions.
    latent_dim = latent.shape[1]
    latent_cols = [f"latent_{i}" for i in range(latent_dim)]

    # Create a new DataFrame with the original BGGId and Name along with latent features.
    df_latent = pd.DataFrame(latent, columns=latent_cols)
    if bggids is not None:
        df_latent.insert(0, "BGGId", bggids)
    if names is not None:
        df_latent.insert(1, "Name", names)

    # Save the latent embeddings to a CSV file.
    df_latent.to_csv(output_csv_path, index=False)
    print(f"Latent embeddings saved to {output_csv_path}")

def calc_store_sim_scores(latent_csv_path, output_csv_path):
    """
    Computes cosine similarity between games using their latent vectors.

    Args:
        latent_csv_path: path to latent embedding CSV (must include BGGId)
        output_csv_path: optional path to save similarity matrix as CSV

    Returns:
        similarity_matrix: pandas DataFrame with index/columns as BGGIds
    """
    sim_df = compute_similarity_matrix(latent_csv_path, output_csv_path)
    sim_df.to_csv(output_csv_path)
    print(f"Saved similarity matrix to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode all games using the trained autoencoder and save latent embeddings to CSV."
    )
    parser.add_argument("--input_csv", type=str, default=GAMES_CSV_PATH,
                        help="Path to the games CSV file.")
    parser.add_argument("--output_csv", type=str, default=GAME_EMBEDDINGS_CSV_PATH,
                        help="Path to the output CSV file with latent embeddings.")
    parser.add_argument("--weights", type=str, default=AUTOENCODER_PATH,
                        help="Path to the trained model weights file.")
    parser.add_argument("--sim_output_csv", type=str, default=GAMES_SIMILARITIES_CSV_PATH,
                        help="Path to save the similarity scores CSV file.")
    args = parser.parse_args()

    encode_all_games(args.input_csv, args.output_csv, args.weights)
    calc_store_sim_scores(args.output_csv, args.sim_output_csv)
