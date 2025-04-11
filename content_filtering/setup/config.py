# data processing parameters
DATA_PARAMS = {
    'continuous_columns_end_index': 514  # first 514 columns are continuous; remaining columns are binary flags
}

# model hyperparameters
MODEL_PARAMS = {
    'latent_dim': 64,
    'encoder_layers': [1028, 512, 128],
    'decoder_layers': [128, 512, 1028],
    'noise_std': 0.2,
}

TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': .0001
}

GAMES_CSV_PATH = "../../bgg_data/overall_games_starter.csv"
GAME_EMBEDDINGS_CSV_PATH = "../../bgg_data/game_embeddings.csv"
GAMES_SIMILARITIES_CSV_PATH = "../../bgg_data/game_similarities.csv"
RATINGS_CSV_PATH = "../../bgg_data/ratings_starter.csv"

AUTOENCODER_PATH = "denoising_autoencoder.pth"
