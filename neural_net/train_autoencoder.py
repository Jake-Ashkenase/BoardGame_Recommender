import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from Autoencoder import Autoencoder, GameDataset

def plot_losses(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('autoencoder_losses.png')
    plt.close()

def main():
    # Load games data
    games_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bgg_data_all/overall_games.csv"))
    games_df = pd.read_csv(games_csv_path)
    games_df = games_df.drop(columns=["Name"], errors='ignore')
    
    # Create dataset
    dataset = GameDataset(games_df)
    
    # Create data loader 
    batch_size = 512  
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Initialize model with larger hidden dimension
    input_dim = games_df.drop(columns=["BGGId"], errors='ignore').shape[1]
    hidden_dim = 256  
    
    print(f"Initializing model with input_dim={input_dim}, hidden_dim={hidden_dim}")
    model = Autoencoder(input_dim, hidden_dim)
    
    # Train model
    print("Starting training...")
    train_losses = model.train_model(
        train_loader=train_loader,
        num_epochs=40,  
        learning_rate=0.0005  
    )
    
    # Plot training curves
    print("Plotting training curves...")
    plot_losses(train_losses)
    
    # Save model
    model_path = 'autoencoder_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 