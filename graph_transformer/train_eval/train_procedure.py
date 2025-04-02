import os
import torch
import torch.nn as nn
import torch.optim as optim
from functools import wraps
from tqdm import tqdm


def track_losses_decorator(train_func):
    """
    Decorator that wraps the 'train' method
    to return lists of train losses and val/test losses over epochs.
    Assumes the trainer internally stores these losses each epoch
    in self._train_losses, self._val_losses, self._test_losses.
    """
    @wraps(train_func)
    def wrapper(self, *args, **kwargs):
        # initialize empty lists to collect losses
        self._train_losses = []
        self._val_losses = []
        self._test_losses = []

        # call original train function
        train_func(self, *args, **kwargs)

        # return the recorded losses
        return self._train_losses, self._val_losses, self._test_losses

    return wrapper


class RatingTrainer:
    """
    A trainer class that handles:
      - Model initialization
      - Training/validation loops
      - Checkpointing (save & load)
      - Basic logging
      - Plotting utilities
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
        device='cpu',
        lr=1e-3,
        checkpoint_dir='./checkpoints',
        checkpoint_prefix='rating_predictor'
    ):
        """
        :param model: The RatingPredictor (or similar) model
        :param train_loader: A PyG LinkNeighborLoader (or similar) for training edges
        :param val_loader: Optional LinkNeighborLoader for validation
        :param test_loader: Optional LinkNeighborLoader for test
        :param device: 'cpu' or 'cuda' (or other device)
        :param lr: learning rate
        :param checkpoint_dir: directory where checkpoint files are stored
        :param checkpoint_prefix: prefix for checkpoint file names
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = checkpoint_prefix

    def train_one_epoch(self):
        """
        One epoch of training over the train_loader.
        Returns the average MSE loss on the training set.
        """
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            # forward pass
            pred_ratings = self.model(batch.x, batch.edge_index, batch.edge_label_index)
            true_ratings = batch.edge_label.view(-1, 1).float()

            loss = self.criterion(pred_ratings, true_ratings)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = true_ratings.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate model on a given loader (val or test).
        Returns the average MSE loss.
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            pred_ratings = self.model(batch.x, batch.edge_index, batch.edge_label_index)
            true_ratings = batch.edge_label.view(-1, 1).float()

            loss = self.criterion(pred_ratings, true_ratings)

            bs = true_ratings.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss

    @track_losses_decorator
    def train(
        self,
        num_epochs=20,
        eval_every=1,
        save_every=1,
        best_val_loss=float('inf')
    ):
        """
        High-level training loop (decorated to return train/val/test losses).
        :param num_epochs: number of epochs to train
        :param eval_every: evaluate on validation set every X epochs
        :param save_every: save checkpoint every X epochs
        :param best_val_loss: pass in previously known best val loss if resuming
        :return: returns (train_losses, val_losses, test_losses) because of the decorator
        """
        # wrap the epoch loop with a tqdm progress bar for overall training
        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Overall Training")
        for epoch in epoch_pbar:
            train_loss = self.train_one_epoch()
            self._train_losses.append(train_loss)

            val_loss = None
            if self.val_loader is not None and (epoch % eval_every == 0):
                val_loss = self.evaluate(self.val_loader)
                self._val_losses.append(val_loss)
            else:
                self._val_losses.append(None)

            test_loss = None
            if self.test_loader is not None:
                test_loss = self.evaluate(self.test_loader)
                self._test_losses.append(test_loss)
            else:
                self._test_losses.append(None)

            # update progress bar
            update_str = f"Train MSE: {train_loss:.4f}"
            if val_loss is not None:
                update_str += f" | Val MSE: {val_loss:.4f}"
            if test_loss is not None:
                update_str += f" | Test MSE: {test_loss:.4f}"
            epoch_pbar.set_postfix_str(update_str)

            # print summary for the epoch
            print(f"Epoch [{epoch}/{num_epochs}] {update_str}")

            # checkpointing
            if (epoch % save_every == 0) and (val_loss is not None):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, best_val_loss, suffix="best")
                    print(f"  => New best model saved! Val MSE: {val_loss:.4f}")

    def save_checkpoint(self, epoch, best_val_loss, suffix="best"):
        """
        Saves model + optimizer state, current epoch, best_val_loss.
        """
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"{self.checkpoint_prefix}_{suffix}.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, ckpt_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads model and optimizer state from a given checkpoint.
        Returns the epoch and best_val_loss from that checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, best_val_loss={best_val_loss:.4f})")
        return epoch, best_val_loss

    @torch.no_grad()
    def get_predictions(self, loader):
        """
        Gather predicted ratings and true ratings for all edges in a given loader.
        :param loader: e.g. val_loader or test_loader
        :return: (list_of_true, list_of_pred)
        """
        self.model.eval()
        all_true = []
        all_pred = []

        for batch in loader:
            batch = batch.to(self.device)
            pred_ratings = self.model(batch.x, batch.edge_index, batch.edge_label_index)
            true_ratings = batch.edge_label.view(-1).float()

            all_true.extend(true_ratings.tolist())
            all_pred.extend(pred_ratings.view(-1).tolist())

        return all_true, all_pred
