import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from prnn import PRNNModel
from crf import CRF


class PRNNCRFModel(nn.Module):
    """A hybrid model using Projection feature based RNN with a CRF layer for tokenized sequence tagging.

    Args:
        num_classes: Number of valid classed for prediction.
        feature_dim: Size of the features from the projection encoder.
        fc_dim: Size of the linear bottleneck layer in the RNN model.
        hidden_dim: Hidden size of the RNN layer(s).
        num_layers: Number of RNN layers to use.
        bidirectional: Whether to set the RNN layer to bidirectional.
        dropout: Dropout probability applied after the bottlneck layer.
        rnn_dropout: Dropout probablity between RNN layers (num_layers > 1).
        crf_rules (optional): A list of transitions and their weights to use when initializing the CRF transition scores.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 128,
        fc_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        rnn_dropout: float = 0.0,
        crf_rules: Optional[List[Tuple[int, int, float]]] = None,
    ):
        super(PRNNCRFModel, self).__init__()
        self.rnn = PRNNModel(
            input_dim=feature_dim,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes + len(CRF.FIXED_CLASSES),
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_dropout=rnn_dropout,
        )
        self.crf = CRF(num_classes, batch_first=True, rules=crf_rules)
        self._config = {
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "fc_dim": fc_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "rnn_dropout": rnn_dropout,
            "crf_rules": crf_rules,
        }

    def forward(self, x, mask=None):
        emissions = self.rnn(x)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, emissions, y, mask=None):
        nll = self.crf(emissions, y, mask=mask)
        return nll

    def _save_config(self, dir: str) -> None:
        with open(f"{dir}/config.json", "w") as config:
            json.dump(self._config, config)

    @staticmethod
    def _load_config(dir: str) -> Dict[str, any]:
        with open(f"{dir}/config.json", "r") as config:
            return json.load(config)

    def save(self, dir: str):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self._save_config(dir)
        torch.save(self.state_dict(), f=f"{dir}/model.pt")

    @classmethod
    def load_from_dir(cls, dir: str):
        config = cls._load_config(dir)
        model = cls(**config)
        state_dict = torch.load(f"{dir}/model.pt")
        model.load_state_dict(state_dict)
        return model
