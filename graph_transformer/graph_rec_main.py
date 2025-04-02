import os
os.environ["MKL_VERBOSE"] = "NO"

from setup import *
from data import *
from models import *
from train_eval import *


def main():

    # build dataset and loaders
    data, train_loader, val_loader, test_loader = get_dataset_and_loaders_for_user_game_link_prediction(
        add_game_game_edges=False,      # or True if you want desc_emb-based edges
        add_shared_attribute_edges=False,  # or True if you want shared mechanic/theme edges
        similarity_threshold=0.75,
        top_k=10,
        val_ratio=0.1,
        test_ratio=0.1,
        num_neighbors=[10, 5],
        batch_size=1024
    )

    print(f"Graph has {data.num_users} user nodes, {data.num_games} game nodes.")
    print(f"x shape: {data.x.shape}, edge_index shape: {data.edge_index.shape}")

    # make model
    in_dim = data.x.size(1)      # dimension of node features
    hidden_dim = 64             # GNN hidden dimension
    out_dim = 64                # final node embedding dimension
    num_heads = 4
    num_layers = 2
    dropout = 0.5

    model = RatingPredictor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=Config.DROPOUT,
        rating_hidden=64,
        rating_out=1,
        rating_layers=2
    )

    # make trainer
    device = Config.DEVICE
    trainer = RatingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=1e-3,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        checkpoint_prefix='rating_transformer'
    )

    # train model
    num_epochs = 10
    train_losses, val_losses, test_losses = trainer.train(
        num_epochs=num_epochs,
        eval_every=1,
        save_every=1
    )

    # plot loss curve
    plot_train_test_losses(
        train_losses,
        val_losses,  # or test_losses if you prefer
        save_as="loss_curve.png",
        title="Train vs. Val MSE Loss"
    )

    # get true and pred for test set
    true_ratings, pred_ratings = trainer.get_predictions(trainer.test_loader)

    # plot true vs pred
    plot_predictions(
        true_ratings,
        pred_ratings,
        save_as="test_predictions.png",
        aggregate=True,
        shaded_region=True,
        smoothing_window=5,
        title="Test Set: Predicted vs. Actual Ratings"
    )

    print("All done!")


if __name__ == "__main__":
    main()
