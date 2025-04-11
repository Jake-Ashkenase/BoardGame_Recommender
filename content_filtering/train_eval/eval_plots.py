import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import pandas as pd


def plot_loss_curves(train_losses, test_losses, save_path="loss_curve.png"):
    """
    Plots the training and test loss curves over epochs.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_embedding_comparison(original_features, latent_features, save_path="embedding_comparison.png"):
    """
    Plots a 2D projection (using PCA) of the original and latent feature spaces side-by-side.
    This helps verify that games that are close in the original feature space remain close
    in the bottleneck embedding space.
    """
    # Reduce original feature dimensions to 2
    pca_orig = PCA(n_components=2)
    original_2d = pca_orig.fit_transform(original_features)

    # Reduce latent feature dimensions to 2
    pca_latent = PCA(n_components=2)
    latent_2d = pca_latent.fit_transform(latent_features)

    # Create side-by-side scatter plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    axs[0].scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.5, c="blue")
    axs[0].set_title("Original Feature Space (PCA)")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")

    axs[1].scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, c="green")
    axs[1].set_title("Latent Embedding Space (PCA)")
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")

    plt.suptitle("Comparison of Original vs. Latent Embedding Space")
    plt.savefig(save_path)
    plt.close()


def plot_neighborhood_preservation(
    originals,
    embeddings,
    game_names=None,
    k=5,
    n_examples=5,
    save_path=None,
    random_seed=42
):
    """
    For each of n_examples, shows only the embedding space with a visual of preserved vs. non-preserved neighbors.
    Each plot includes a subtitle with the numeric neighbor overlap score (e.g., 3/5 preserved).
    """
    np.random.seed(random_seed)
    indices = np.random.choice(len(originals), size=n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, 2)

    orig_dists = euclidean_distances(originals)
    emb_dists = euclidean_distances(embeddings)

    for i, idx in enumerate(indices):
        orig_neighbors = np.argsort(orig_dists[idx])[1:k + 1]
        emb_neighbors = np.argsort(emb_dists[idx])[1:k + 1]

        combined_indices = [idx] + list(set(orig_neighbors).union(set(emb_neighbors)))
        orig_sub = originals[combined_indices]
        emb_sub = embeddings[combined_indices]

        orig_2d = PCA(n_components=2).fit_transform(orig_sub)
        emb_2d = PCA(n_components=2).fit_transform(emb_sub)

        def plot_sub(ax, coords, neighbors, space_name):
            ax.scatter(coords[1:, 0], coords[1:, 1], color='gray', label='Other Neighbors', alpha=0.5)
            for j in range(1, len(coords)):
                if j - 1 in neighbors:
                    ax.scatter(coords[j, 0], coords[j, 1], color='blue', label='Preserved Neighbor' if j == 1 else "")
            ax.scatter(coords[0, 0], coords[0, 1], color='red', label='Target Game', edgecolor='black', s=80)

            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title(f"{space_name} (Game {idx})" + (f": {game_names[idx]}" if game_names else ""))
            ax.grid(True)
            ax.legend(loc='best')

        #plot_sub(axes[i, 0], orig_2d, range(k), "Original Feature Space")
        plot_sub(axes[i], emb_2d, range(k), "Embedding Space")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(true, pred, save_as=None, aggregate=True, shaded_region=True, smoothing_window=5,
                     title="Predicted vs. Actual"):
    """
    Plot a scatter or aggregated means/min/max region of predicted vs actual.
    :param true: list or array of actual values
    :param pred: list or array of predicted values
    :param save_as: optional filepath to save figure
    :param aggregate: if True, group predictions by actual rating and show mean
    :param shaded_region: if True, also fill between min & max predictions
    :param smoothing_window: rolling window for smoothing
    :param title: plot title
    """
    true = np.array(true).flatten()
    pred = np.array(pred).flatten()

    plt.figure(figsize=(8, 6), facecolor='#384957')

    if aggregate or shaded_region:
        df = pd.DataFrame({"actual": true, "predicted": pred})
        grouped = df.groupby("actual")["predicted"]
        mean_pred = grouped.mean()
        min_pred = grouped.min()
        max_pred = grouped.max()
        sorted_actual = mean_pred.index

        # rolling smoothing
        smoothed_mean = mean_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smoothed_min = min_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smoothed_max = max_pred.rolling(window=smoothing_window, center=True, min_periods=1).mean()

        if shaded_region:
            plt.fill_between(sorted_actual, smoothed_min.values, smoothed_max.values,
                             color="#FF6B65", alpha=0.15, label="Prediction Range")
        if aggregate:
            plt.plot(sorted_actual, smoothed_mean.values, color="#FF6B65", label="Mean Prediction")
    else:
        # just scatter plot
        plt.scatter(true, pred, alpha=0.5, label="Predicted vs Actual", color="#FF6B65", s=5)

    # perfect prediction line
    tmin, tmax = min(true), max(true)
    min_val = min(tmin, 0)
    max_val = max(tmax, 10)
    plt.plot([min_val, max_val], [min_val, max_val],
             '--', label="Perfect Prediction", color="#384957")

    plt.xlabel("Actual", color="white")
    plt.ylabel("Predicted", color="white")
    plt.title(title, color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    plt.grid(True)

    if save_as:
        plt.savefig(save_as)
    plt.show()
