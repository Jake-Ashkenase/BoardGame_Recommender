import os
import torch
import torch.nn as nn
import torch.optim as optim
from functools import wraps
from tqdm import tqdm
import multiprocessing
from .eval_plots import async_predictions_plot, async_gradient_flow_plot
from setup import *


def track_losses_decorator(train_func):
    """
    Decorator that wraps the 'train' method
    to return lists of train losses and val/test losses over epochs.
    Assumes the trainer internally stores these losses each epoch
    in self._train_losses, self._val_losses, self._test_losses.
    """
    @wraps(train_func)
    def wrapper(self, *args, **kwargs):
        self._train_losses = []
        self._val_losses = []
        self._test_losses = []
        train_func(self, *args, **kwargs)
        return self._train_losses, self._val_losses, self._test_losses
    return wrapper

def checkpoint_and_plot_decorator(save_func):
    @wraps(save_func)
    def wrapper(self, epoch, best_val_loss, suffix="best"):
        save_func(self, epoch, best_val_loss, suffix)
        if hasattr(self, "_cached_train_true") and hasattr(self, "_cached_train_pred"):
            train_true = self._cached_train_true
            train_pred = self._cached_train_pred
        else:
            train_true, train_pred = self.get_predictions(self.train_loader)
        if self.test_loader is not None:
            test_true, test_pred = self.get_predictions(self.test_loader)
        else:
            test_true, test_pred = [], []
        train_plot_path = os.path.join(self.checkpoint_dir, f"test_predictions_epoch{epoch}.png")
        test_plot_path = os.path.join(self.checkpoint_dir, f"test_real_predictions_epoch{epoch}.png")
        p = multiprocessing.Process(
            target=async_predictions_plot,
            args=(epoch, train_true, train_pred, test_true, test_pred,
                  train_plot_path, test_plot_path, self.block_plot)
        )
        p.start()
        if self.block_plot:
            p.join()
    return wrapper

class RatingTrainer:
    """
    A trainer class that handles:
      - Model initialization
      - Training/validation loops
      - Checkpointing (save & load)
      - Basic logging
      - Plotting utilities including gradient flow plots.
    """
    def __init__(
            self,
            model,
            train_loader,
            val_loader=None,
            test_loader=None,
            criterion=None,
            device='cpu',
            transformer_lr=.001,
            mpl_lr=.001,
            checkpoint_dir='./checkpoints',
            checkpoint_prefix='rating_predictor',
            weight_decay=.00001,
            block_plot=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion.to(self.device)

        gnn_params = []
        mlp_params = []
        for name, param in model.named_parameters():
            if "gnn." in name:  # belongs to GraphTransformer
                gnn_params.append(param)
            else:  # belongs to EdgeMLP or embeddings
                mlp_params.append(param)

        self.optimizer = optim.Adam([
            {"params": gnn_params, "lr": transformer_lr},  # higher LR for GNN
            {"params": mlp_params, "lr": mpl_lr}  # lower LR for MLP
        ], weight_decay=weight_decay)

        # Instantiate a learning rate scheduler that reduces LR on plateau (based on validation loss)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True
        )

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = checkpoint_prefix
        self.block_plot = block_plot

        # Register gradient hooks for monitoring gradient flow.
        self.register_gradient_hooks()

    def register_gradient_hooks(self):
        """Register a hook on each trainable parameter to record its gradient norm on each backward pass."""
        self._gradients_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, n=name: self._save_gradient(n, grad))

    def _save_gradient(self, name, grad):
        norm = grad.norm().item()
        if name not in self._gradients_dict:
            self._gradients_dict[name] = []
        self._gradients_dict[name].append(norm)

    def _forward_pass(self, batch):
        batch = batch.to(self.device)
        if hasattr(batch, 'n_id'):
            return self.model(batch.x, batch.edge_index, batch.edge_label_index, n_id=batch.n_id)
        else:
            return self.model(batch.x, batch.edge_index, batch.edge_label_index)

    def train_one_epoch(self, return_predictions=False):
        self.model.train()
        total_loss = 0
        total_samples = 0
        cached_true = []
        cached_pred = []
        for batch in self.train_loader:
            pred_ratings = self._forward_pass(batch)
            true_ratings = batch.edge_label.view(-1, 1).float()
            loss = self.criterion(pred_ratings, true_ratings)
            self.optimizer.zero_grad()
            loss.backward()
            # clip gradient to fix uneven gradients across layers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            bs = true_ratings.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            if return_predictions:
                cached_true.extend(true_ratings.tolist())
                cached_pred.extend(pred_ratings.view(-1).tolist())
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        if return_predictions:
            return avg_loss, cached_true, cached_pred
        return avg_loss

    @torch.inference_mode()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        for batch in loader:
            pred_ratings = self._forward_pass(batch)
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
            best_val_loss=float('inf'),
            early_stopping_patience=3
    ):
        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Overall Training")
        patience_counter = 0
        for epoch in epoch_pbar:
            self._gradients_dict = {}  # Reset gradient storage for the new epoch.
            train_loss, cached_true, cached_pred = self.train_one_epoch(return_predictions=True)
            self._train_losses.append(train_loss)
            self._cached_train_true = cached_true
            self._cached_train_pred = cached_pred

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

            update_str = f"Train MSE: {train_loss:.4f}"
            if val_loss is not None:
                update_str += f" | Val MSE: {val_loss:.4f}"
            if test_loss is not None:
                update_str += f" | Test MSE: {test_loss:.4f}"
            epoch_pbar.set_postfix_str(update_str)

            # Update learning rate scheduler with validation loss.
            if val_loss is not None:
                self.scheduler.step(val_loss)

            # Compute average gradient norms per parameter for this epoch.
            avg_grad_flow = {name: sum(vals) / len(vals) if len(vals) > 0 else 0.0
                             for name, vals in self._gradients_dict.items()}

            # Launch asynchronous gradient flow plot.
            grad_plot_path = os.path.join(self.checkpoint_dir, f"grad_flow_epoch{epoch}.png")
            p_grad = multiprocessing.Process(
                target=async_gradient_flow_plot,
                args=(epoch, avg_grad_flow, grad_plot_path, self.block_plot)
            )
            p_grad.start()

            # Checkpointing and early stopping based on validation loss.
            if val_loss is None:
                val_loss = train_loss
            # if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, best_val_loss, suffix="best")
                print(f"  => New best model saved! Val MSE: {val_loss:.4f}")
            else:
                self.save_checkpoint(epoch, val_loss, suffix="current")
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    @checkpoint_and_plot_decorator
    def save_checkpoint(self, epoch, best_val_loss, suffix="best"):
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
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, best_val_loss={best_val_loss:.4f})")
        return epoch, best_val_loss

    @torch.inference_mode()
    def get_predictions(self, loader):
        self.model.eval()
        all_true = []
        all_pred = []
        for batch in loader:
            pred_ratings = self._forward_pass(batch)
            true_ratings = batch.edge_label.view(-1).float()
            all_true.extend(true_ratings.tolist())
            all_pred.extend(pred_ratings.view(-1).tolist())
        return all_true, all_pred
