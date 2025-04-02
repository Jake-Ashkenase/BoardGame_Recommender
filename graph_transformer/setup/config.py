import os


class Config:
    """
    Configuration settings for the board game recommender system with graph transformer.
    Contains hyperparameters, file paths, and other options.
    """

    PROJECT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..", "..")
    )

    # The directory containing this config.py: "graph_transformer"
    GT_DIR = os.path.abspath(os.path.dirname(__file__))

    # bgg_data folder (alongside BoardGame_Recommender in "Final")
    DATA_DIR = os.path.join(PROJECT_DIR, "bgg_data")

    # Local checkpoints folder inside graph_transformer
    CHECKPOINT_DIR = os.path.join(GT_DIR, "../checkpoints")

    # Data Files
    BGG_DATA_FILE = os.path.join(DATA_DIR, "bgg_data_documentation.txt")
    USER_RATINGS_FILE = os.path.join(DATA_DIR, "ratings_filtered.csv")
    OVERALL_GAMES_FILE = os.path.join(DATA_DIR, "overall_games.csv")

    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    DROPOUT = 0.5
    NUM_TRANSFORMER_LAYERS = 4
    EMBEDDING_DIM = 128
    NUM_HEADS = 8

    # Training Settings
    LOG_INTERVAL = 10  # iterations
    SEED = 42

    # Device Configuration
    # Example logic: if CUDA_VISIBLE_DEVICES is set, use 'cuda', else try 'mps' as fallback
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    @classmethod
    def ensure_directories(cls):
        """Ensure that any necessary directories exist (e.g., for saving checkpoints)."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)


# Run this to ensure directories exist when config is imported.
Config.ensure_directories()

