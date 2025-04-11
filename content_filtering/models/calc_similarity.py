# models/compute_similarity.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def standardize_row(row):
    row_min = row.min()
    row_max = row.max()
    if row_max - row_min == 0:
        return row  # Avoid division by zero if row is constant.
    return (row - row_min) / (row_max - row_min)


def compute_similarity_matrix(latent_csv_path, output_csv_path=None):
    """
    Computes cosine similarity between games using their latent vectors.

    Args:
        latent_csv_path: path to latent embedding CSV (must include BGGId)
        output_csv_path: optional path to save similarity matrix as CSV

    Returns:
        similarity_matrix: pandas DataFrame with index/columns as BGGIds
    """
    df = pd.read_csv(latent_csv_path)
    bgg_ids = df['BGGId']
    features = df.drop(columns=['BGGId', 'Name'], errors='ignore').values

    similarity = cosine_similarity(features)
    sim_df = pd.DataFrame(similarity, index=bgg_ids, columns=bgg_ids)

    sim_df = sim_df.apply(standardize_row, axis=1)

    if output_csv_path:
        sim_df.to_csv(output_csv_path)
        print(f"Saved similarity matrix to {output_csv_path}")

    return sim_df


