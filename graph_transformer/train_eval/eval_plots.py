import matplotlib.pyplot as plt
import pandas as pd


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
    