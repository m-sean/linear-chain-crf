from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from convolve.feature.projection import ProjectionEncoder
from convolve.distortion import TextDistorter
from torch.utils.data import Dataset

from utils import pad_sequence
from crf import CRF


class TokenizedDataset(Dataset):
    """A pytorch-compatible dataset for training sequence tagging models.
    """

    def __init__(self, x, y, masks) -> None:
        super(TokenizedDataset, self).__init__()
        self.x = x
        self.y = y
        self.masks = masks

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.masks[idx]

    @classmethod
    def build(
        cls,
        data: str,
        encoder: ProjectionEncoder,
        tag_to_idx: Dict[str, int],
        distorter: Optional[TextDistorter] = None,
        fit_tags: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        df = pd.read_csv(data)
        x, y, masks = _get_sequences(
            df, encoder, distorter, tag_to_idx, fit_tags, device
        )
        return cls(x, y, masks)


def _encode_text(
    text: List[str],
    encoder: ProjectionEncoder,
    distorter: TextDistorter,
    device: torch.device,
):
    enc = encoder.encode(text, distorter)
    return torch.tensor(enc, dtype=torch.float, device=device)


def _pad_labels(labels: List[int], max_len: int, pad_val: any, device: torch.device):
    labels = pad_sequence(labels, max_len, pad_val)
    return torch.tensor(labels, dtype=torch.long, device=device)


def _get_sequences(
    dataframe: pd.DataFrame,
    encoder: ProjectionEncoder,
    distorter: TextDistorter,
    tag_to_idx: Dict[str, int],
    fit_tags: bool,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    x = []
    y = []
    masks = []
    x_tmp = []
    # <BOS> and <EOS> tags are not valid predictions, so we omit them with padding
    y_tmp = [
        0,
    ]
    for row in dataframe.itertuples():
        token = str(row.token)
        tag = row.tag
        if token == "-EOS-":
            if not x_tmp:
                continue
            # encode text (adds <BOS> and <EOS> encodings and padding)
            x_tensor = _encode_text(x_tmp, encoder, distorter, device)
            # add label padding
            y_tensor = _pad_labels(y_tmp, x_tensor.shape[0], 0, device)
            mask_tensor = torch.ne(y_tensor, 0).float()
            x.append(x_tensor)
            y.append(y_tensor)
            masks.append(mask_tensor)
            x_tmp = []
            y_tmp = [
                0,
            ]
            continue
        if fit_tags and tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx) + len(CRF.FIXED_CLASSES)
        x_tmp.append(token)
        y_tmp.append(tag_to_idx[tag])
    return x, y, masks
