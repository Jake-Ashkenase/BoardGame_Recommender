import os
os.environ["MKL_VERBOSE"] = "NO"

from setup import *
from data import *
from models import *
from train_eval import *
import matplotlib.pyplot as plt
import torch
from loss_functions import *




def get_inverse_frequency_dict(train_loader, freq_factor=.5):
    from collections import defaultdict
    rating_counts = defaultdict(int)

    for batch in train_loader:
        # batch.edge_label holds the ground-truth ratings for that batch
        ratings = batch.edge_label.view(-1).tolist()
        for r in ratings:
            r_rounded = round(r, 2)
            rating_counts[r_rounded] += 1

    inverse_freqs = {rating: 1 / (count ** freq_factor) for rating, count in rating_counts.items()}
    return inverse_freqs


def main():
    print("Using device:", Config.DEVICE)
    # build dataset and loaders
    (data,
     train_loader, val_loader, test_loader,
     ug_src, ug_dst, ug_attr,
     train_idx, val_idx, test_idx
    ) = get_dataset_and_loaders_for_user_game_link_prediction(
        add_game_game_edges=True,
        add_shared_attribute_edges=True,
        similarity_threshold=0.75,
        top_k=10,
        val_ratio=.1,
        test_ratio=.1,
        num_neighbors=[10, 5],
        batch_size=1024
    )

    print(f"Graph has {data.num_users} user nodes, {data.num_games} game nodes.")
    print(f"x shape: {data.x.shape}, edge_index shape: {data.edge_index.shape}")

    train_ug_count = train_idx.size(0)
    val_ug_count = val_idx.size(0)
    test_ug_count = test_idx.size(0)
    print(f"Train user->game edges: {train_ug_count}")
    print(f"Val user->game edges:   {val_ug_count}")
    print(f"Test user->game edges:  {test_ug_count}")

    # plot rating distributions for train & test sets
    # train_ratings = ug_attr[train_idx]  # [train_ug_count]
    # test_ratings = ug_attr[test_idx]    # [test_ug_count]

    # plot_rating_hist(train_ratings, title="Train Rating Distribution", bins=20, save_as="train_rating_dist.png")
    # plot_rating_hist(test_ratings, title="Test Rating Distribution", bins=20, save_as="test_rating_dist.png")

    # make model
    in_dim = data.x.size(1)  # dimension of node features
    model = RatingPredictor(
        in_dim=in_dim,
        hidden_dim=Config.EMBEDDING_DIM_TRANSFORMER,
        out_dim=Config.EMBEDDING_DIM_MLP,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_TRANSFORMER_LAYERS,
        dropout=Config.DROPOUT,
        rating_hidden=Config.EMBEDDING_DIM_MLP,
        rating_out=1,
        rating_layers=Config.NUM_MLP_LAYERS,
        num_users=data.num_users,
        num_games=data.num_games
    )
    # state_dict = torch.load(Config.CHECKPOINT_DIR + "/rating_transformer_best.pth")
    # model.load_state_dict(state_dict['model_state_dict'])

    # setup custom loss
    rating_to_weight = get_inverse_frequency_dict(train_loader, freq_factor=1)
    criterion = InverseFrequencyMSELoss(rating_to_weight)

    # make trainer
    trainer = RatingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=Config.DEVICE,
        mpl_lr=Config.MLP_LEARNING_RATE,
        transformer_lr=Config.TRANSFORMER_LEARNING_RATE,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        checkpoint_prefix='rating_transformer',
        weight_decay=Config.WEIGHT_DECAY
    )

    # train model
    train_losses, val_losses, test_losses = trainer.train(
        num_epochs=Config.NUM_EPOCHS,
        eval_every=Config.EVAL_EVERY,
        save_every=Config.SAVE_EVERY,
        early_stopping_patience=Config.PATIENCE
    )

    # plot loss curve
    plot_train_test_losses(
        train_losses,
        val_losses,
        save_as="loss_curve.png",
        title="Train vs. Val MSE Loss"
    )

    # evaluate on test and train sets and plot predictions
    true_ratings_test, pred_ratings_test = trainer.get_predictions(trainer.test_loader)
    plot_predictions(
        true_ratings_test,
        pred_ratings_test,
        save_as="test_predictions.png",
        aggregate=True,
        shaded_region=True,
        smoothing_window=20,
        title="Test Set: Predicted vs. Actual Ratings"
    )
    true_ratings_test, pred_ratings_test = trainer.get_predictions(trainer.train_loader)
    plot_predictions(
        true_ratings_test,
        pred_ratings_test,
        save_as="train_predictions.png",
        aggregate=True,
        shaded_region=True,
        smoothing_window=20,
        title="Train Set: Predicted vs. Actual Ratings"
    )

    print("All done!")


if __name__ == "__main__":
    main()
