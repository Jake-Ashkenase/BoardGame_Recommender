import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_train_test_losses(train_losses, test_losses, save_as=None, title="Train vs. Test Loss"):
    """
    Plot train and test (or val) losses over epochs.
    Takes lists of losses of the same length or test can be shorter if some epochs are missing.
    """
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    # get rid of None vals for if not evaluated at every epoch
    if any(x is not None for x in test_losses):
        valid_indices = [(i, x) for i, x in enumerate(test_losses, start=1) if x is not None]
        if valid_indices:
            test_epochs, test_values = zip(*valid_indices)
            plt.plot(test_epochs, test_values, label="Test Loss", color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_as:
        plt.savefig(save_as)
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


def plot_rating_hist(ratings, title="Rating Distribution", bins=20, save_as=None):
    """
    Simple histogram for rating distribution.
    """
    plt.figure(figsize=(7, 5))
    plt.hist(ratings.cpu().numpy(), bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    if save_as:
        plt.savefig(save_as)
    plt.show()


def async_gradient_flow_plot(epoch, grad_flow, grad_plot_path, block_plot=False):
    """
    Asynchronously plot the gradient flow.
    :param epoch: Current epoch number.
    :param grad_flow: Dictionary mapping parameter names to average gradient norms.
    :param grad_plot_path: File path to save the plot.
    :param block_plot: If True, display the plot.
    """
    plt.figure(figsize=(10, 6))
    layers = list(grad_flow.keys())
    avg_grad = list(grad_flow.values())
    plt.bar(range(len(layers)), avg_grad, color="blue")
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Average Gradient Norm")
    plt.title(f"Gradient Flow at Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(grad_plot_path)
    if block_plot:
        plt.show()
    plt.close()


def async_predictions_plot(epoch, train_true, train_pred, test_true, test_pred,
                           train_plot_path, test_plot_path, block_plot=False):
    """
    Helper function to be run in a separate process.
    It calls plot_predictions for both train and test sets.
    If block_plot is True, it will block waiting for user interaction.
    """
    # Plot training predictions
    plot_predictions(
        true=train_true,
        pred=train_pred,
        save_as=train_plot_path,
        aggregate=True,
        shaded_region=True,
        smoothing_window=20,
        title=f"Train Set: Predicted vs. Actual Ratings (Epoch {epoch})"
    )
    if len(test_true) > 0:
        # Plot test predictions
        plot_predictions(
            true=test_true,
            pred=test_pred,
            save_as=test_plot_path,
            aggregate=True,
            shaded_region=True,
            smoothing_window=20,
            title=f"Test Set: Predicted vs. Actual Ratings (Epoch {epoch})"
        )
    if block_plot:
        plt.show()
