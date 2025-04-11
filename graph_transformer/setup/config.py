import os


class Config:
    """
    Configuration settings for the board game recommender system with graph transformer.
    Contains hyperparameters, file paths, and other options.
    """

    PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/Final"
    GT_DIR = os.path.join(PROJECT_DIR, "BoardGame_Recommender")
    DATA_DIR = os.path.join(PROJECT_DIR, "bgg_data")
    CHECKPOINT_DIR = os.path.join(GT_DIR, "checkpoints")

    USER_RATINGS_FILE = os.path.join(DATA_DIR, "ratings_starter.csv")
    OVERALL_GAMES_FILE = os.path.join(DATA_DIR, "overall_games_starter.csv")

    # eval and checkpoint patterns
    EVAL_EVERY = 1
    SAVE_EVERY = 1

    # Hyperparameters
    MLP_LEARNING_RATE = 0.001
    TRANSFORMER_LEARNING_RATE = 0.008
    BATCH_SIZE = 256
    NUM_EPOCHS = 20

    # model capacity
    NUM_TRANSFORMER_LAYERS = 3
    EMBEDDING_DIM_TRANSFORMER = 256
    EMBEDDING_DIM_MLP = 128  # reduced from 256 to lower capacity
    INPUT_DIM_MLP = 256
    NUM_MLP_LAYERS = 3  # reduced from 4
    NUM_HEADS = 8

    # hyperparameters fo regularization and scheduling
    EDGE_DROPOUT = 0.2  # Fraction of edges to drop in the GNN
    SCHEDULER_FACTOR = 0.5  # Factor for reducing the learning rate on plateau
    SCHEDULER_PATIENCE = 2  # Number of epochs with no improvement before reducing LR
    DROPOUT = 0.4  # was .5 for no overfitting
    WEIGHT_DECAY = .00005  # increased from .00001 for overfitting

    # Training Settings
    LOG_INTERVAL = 10  # iterations
    SEED = 20
    PATIENCE = 5

    # Device Configuration
    DEVICE = 'cuda'  # if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'

    BACKGROUND_COLOR = "#faf9f6"
    PRIMARY_COLOR = "#f59a1b"
    SECONDARY_COLOR = "#f52e3c"
    TERNARY_COLOR = "#f5cdbf"
    QUATERNARY_COLOR = "#100e0e"

    @classmethod
    def ensure_directories(cls):
        """Ensure that any necessary directories exist (e.g., for saving checkpoints)."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)


# Run this to ensure directories exist when config is imported.
Config.ensure_directories()
