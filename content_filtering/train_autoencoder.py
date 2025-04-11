# models/train_autoencoder.py

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import *
from setup import *
from train_eval import *



def load_dataset(csv_path):
    """
    Loads the game features dataset from CSV.
    Assumes all columns are numeric and preprocessed.
    """
    df = pd.read_csv(csv_path)
    # drop "BGGId" and "Name"
    df = df.drop(columns=['BGGId', 'Name'], errors='ignore')
    data = df.values.astype(np.float32)
    return data


def create_dataloaders(data, batch_size=TRAINING_PARAMS['batch_size'], test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets and creates DataLoaders.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    test_dataset = TensorDataset(torch.from_numpy(test_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def extract_embeddings_from_loader(model, data_loader, device):
    """
    Extracts latent embeddings and original inputs using the model's built-in encode method.
    """
    model.eval()
    embeddings = []
    originals = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            latent = model.encode(inputs)
            embeddings.append(latent.cpu().numpy())
            originals.append(inputs.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    originals = np.concatenate(originals, axis=0)
    return originals, embeddings


def run_autoencoder_training(csv_path):
    torch.manual_seed(42)
    np.random.seed(42)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Training on device: {device}")

    # Load dataset
    data = load_dataset(csv_path)

    # Create train and test DataLoaders
    train_loader, test_loader = create_dataloaders(data)

    # Define the autoencoder model
    input_dim = data.shape[1]
    model = build_autoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])

    # Train using the generic training module
    train_losses, test_losses = train_model_generic(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_func=MSE_BCE_Loss(split_index=514),
        device=device,
        epochs=TRAINING_PARAMS['epochs']
    )

    # Save loss curves and model weights
    plot_loss_curves(train_losses, test_losses, save_path="loss_curve.png")
    torch.save(model.state_dict(), "denoising_autoencoder.pth")

    # Extract latent embeddings from the test set using model.encode()
    originals, latent_embeddings = extract_embeddings_from_loader(model, test_loader, device)
    plot_embedding_comparison(originals, latent_embeddings, save_path="embedding_comparison.png")

    plot_neighborhood_preservation(originals, latent_embeddings, k=10, n_examples=3,
                                   save_path="neighborhood_preservation.png")

    # Additional Evaluation Plots:
    # plot_tsne_latent(latent_embeddings, save_path="tsne_latent.png")

    # Obtain reconstructions for the entire test set (using the full forward pass)
    # model.eval()
    # reconstructions = []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         inputs = batch[0].to(device)
    #         outputs, _ = model(inputs)
    #         reconstructions.append(outputs.cpu().numpy())
    # reconstructions = np.concatenate(reconstructions, axis=0)
    # plot_reconstruction_error(originals, reconstructions, save_path="reconstruction_error_hist.png")

    # Plot example reconstructions for a few random samples
    # sample_indices = np.random.choice(originals.shape[0], size=5, replace=False)
    # plot_sample_reconstructions(originals, reconstructions, sample_indices, save_path="sample_reconstructions.png")


if __name__ == "__main__":
    run_autoencoder_training(GAMES_CSV_PATH)
