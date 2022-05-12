import logging

from tqdm import tqdm
from typing import Optional

import torch

from convolve.distortion import TextDistorter
from convolve.feature.projection import ProjectionEncoder
from torch.utils.data import DataLoader

from data import TokenizedDataset
from prnn_crf import PRNNCRFModel
from optimizer import OptimizerConfig
from utils import masked_acc


class PRNNCRFTrainer:
    """Trains a Projection feature based hybrid RNN-CRF model.

    Args:
        encoder: A convolve ProjectionEncoder.
        distorter: A convolve TextDistorter for the training set.
        batch_size: Input batch size.
    """

    def __init__(
        self,
        encoder: ProjectionEncoder,
        distorter: Optional[TextDistorter] = None,
        batch_size: int = 128,
    ) -> None:
        self.encoder = encoder
        self.distorter = distorter
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.history = None
        self.tag_to_idx = None

    def num_classes(self):
        if not self.tag_to_idx:
            raise AttributeError("Tags have not yet been fit to a training set")
        return len(self.tag_to_idx)

    def get_idx_to_tag(self):
        if not self.tag_to_idx:
            raise AttributeError("Tags have not yet been fit to a training set")
        return {idx: tag for tag, idx in self.tag_to_idx}

    def get_data_loader(self, data: str, train: bool, shuffle: bool = True):
        """Loads training and validation sets from CSV. See TokenizedDataset for data requirements."""
        if train:
            if self.tag_to_idx is not None:
                dataset = TokenizedDataset.build(
                    data,
                    self.encoder,
                    self.tag_to_idx,
                    distorter=self.distorter,
                    device=self.device,
                )
                return DataLoader(dataset, self.batch_size, shuffle)
            self.tag_to_idx = {}
            dataset = TokenizedDataset.build(
                data,
                self.encoder,
                self.tag_to_idx,
                distorter=self.distorter,
                fit_tags=True,
                device=self.device,
            )
            return DataLoader(dataset, self.batch_size, shuffle)
        if self.tag_to_idx is None:
            raise AttributeError("Tags have not yet been fit to a training set")
        dataset = TokenizedDataset.build(
            data, self.encoder, self.tag_to_idx, device=self.device
        )
        return DataLoader(dataset, self.batch_size, shuffle)

    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        epochs: int,
        save: str,
        optimizer: OptimizerConfig,
        **model_kwargs,
    ) -> PRNNCRFModel:
        """Performs model training with early stopping after no improvement in validation loss after 3 epochs.
        
        Args:
            train: Training set Dataloader.
            val: Validation set Dataloader.
            save: Location to save the best model from training.
            optimizer: Optimizer configuration.
            model_kwargs: See PRNNCRFModel for options.
        """
        model = PRNNCRFModel(**model_kwargs)
        self.optimizer = optimizer.get_optimizer(model.parameters())
        model = model.to(device=self.device)
        self.history = {"train": [], "val": []}
        batch_ct = len(train)
        for e in range(1, epochs + 1):
            with tqdm(total=batch_ct, unit="batch") as pbar:
                pbar.set_description(f"Epoch {e}")
                train_acc, train_loss = self._train_step(model, train, pbar)
                val_acc, val_loss = self._val_step(model, val)
                pbar.set_postfix_str(
                    f"train_acc={train_acc:.4f}, train_loss={train_loss:.2f},"
                    f"val_acc={val_acc:.4f}, val_loss={val_loss:.2f}"
                )
            self.history["train"].append((train_acc, train_loss))
            self.history["val"].append((val_acc, val_loss))
            if e == 1 or val_loss < best_loss:
                logging.info(f"Saving model to {save}/model.pt")
                best_epoch = e
                best_loss = val_loss
                model.save(save)

            elif e - best_epoch > 2:
                logging.info("No improvement after 3 epochs, stopping early.")
                break
        return model

    def evaluate(self, model: PRNNCRFModel, eval: DataLoader):
        logging.info("Running model evaluation")
        batch_ct = len(eval)
        with tqdm(total=batch_ct, unit="batch") as pbar:
            pbar.set_description("Model evaluation")
            eval_acc, eval_loss = self._val_step(model, eval, pbar)
            pbar.set_postfix_str(f"ACC={eval_acc:.4f}, LOSS={eval_loss:.2f}")
        return (eval_acc, eval_loss)

    def _train_step(self, model: PRNNCRFModel, train: DataLoader, pbar: tqdm):
        sum_loss = 0.0
        sum_acc = 0.0
        model = model.train()
        for i, (x, y_true, mask) in enumerate(train):
            for p in model.parameters():
                p.grad = None
            emissions = model.rnn(x)
            # get loss and extract metric
            loss = model.loss(emissions, y_true, mask=mask)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.detach()
            avg_loss = sum_loss / (i + 1)
            # get acc metric
            _, y_pred = model.crf.decode(emissions, mask=mask)
            sum_acc += masked_acc(y_pred, y_true, mask)
            avg_acc = sum_acc / (i + 1)
            pbar.update()
            pbar.set_postfix_str(f"acc={avg_acc:.4f}, loss={avg_loss:.2f}")
        return avg_acc, avg_loss

    def _val_step(self, model: PRNNCRFModel, val: DataLoader, pbar=None):
        sum_loss = 0.0
        sum_acc = 0.0
        with torch.no_grad():
            model = model.eval()
            for i, (x, y_true, mask) in enumerate(val):
                emissions = model.rnn(x)
                loss = model.loss(emissions, y_true, mask=mask)
                sum_loss += loss.detach()
                avg_loss = sum_loss / (i + 1)
                _, y_pred = model.crf.decode(emissions, mask=mask)
                sum_acc += masked_acc(y_pred, y_true, mask)
                avg_acc = sum_acc / (i + 1)
                if pbar:
                    pbar.update()
        return avg_acc, avg_loss
