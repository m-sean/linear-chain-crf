from typing import Dict, List, Tuple

import pandas as pd
import torch

from crf import CRF


def masked_acc(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> float:
    pred_ct = mask.sum().detach()
    pad_ct = torch.numel(y_pred) - pred_ct
    correct = (y_pred == y_true).sum().detach() - pad_ct
    return correct / pred_ct


def pad_sequence(sequence: List[any], max_len: int, pad_val: any) -> List[any]:
    for _ in range(max_len - len(sequence)):
        sequence.append(pad_val)
    return sequence


def get_ner_rules(tag_to_idx: Dict[str, int]) -> List[Tuple[int, int, float]]:
    """Helps to rule out impossible transitions during training by generating a list 
    of transition scores used by the CRF to initalize transition weights.
    """
    pad_idx = CRF.FIXED_CLASSES["<PAD>"]
    bos_idx = CRF.FIXED_CLASSES["<BOS>"]
    eos_idx = CRF.FIXED_CLASSES["<EOS>"]
    o_idx = tag_to_idx["O"]
    rules = [
        (eos_idx, pad_idx, 0.0),
        (pad_idx, pad_idx, 0.0),
    ]
    non_ent_idx = {bos_idx, eos_idx, pad_idx, o_idx}
    for tag, idx in tag_to_idx.items():
        # no transitions allowed to the beginning of sentence
        rules.append((idx, bos_idx, -10000.0))
        # no transition allowed from the end of sentence unless it's <PAD>
        if tag == "<PAD>":
            continue
        rules.append((eos_idx, idx, -10000.0))
        # no transitions from padding
        rules.append((pad_idx, idx, -10000.0))
        if idx in non_ent_idx:
            continue
        bio_tag, ent_tag = tag.split("-")
        # No transitions from <BOS> or O to I-*:
        if bio_tag == "I":
            rules.append((bos_idx, idx, -10000.0))
            rules.append((o_idx, idx, -10000.0))
        # No trainsitons from B-X to I-X'
        for to_tag, to_idx in tag_to_idx.items():
            if not to_tag.startswith("I"):
                continue
            _, ent_tag_to = to_tag.split("-")
            if bio_tag == "B" and ent_tag != ent_tag_to:
                rules.append((idx, to_idx, -10000.0))
    return rules


def get_text_sequences(file, max_seq_len):
    df = pd.read_csv(file)
    x = []
    y = []
    x_tmp = [
        "<PAD>",
    ]
    # <BOS> and <EOS> tags are not valid predictions, so we omit them with padding
    y_tmp = [
        "<PAD>",
    ]

    for row in df.itertuples():
        token = str(row.token)
        if token == "-EOS-":
            if not x_tmp:
                continue
            # encode text (adds <BOS> and <EOS> encodings and padding)
            x_tmp = pad_sequence(x_tmp, max_seq_len, "<PAD>")
            # add tag padding
            y_tmp = pad_sequence(y_tmp, max_seq_len, "<PAD>")
            x.append(x_tmp)
            y.append(y_tmp)
            x_tmp = [
                "<PAD>",
            ]
            y_tmp = [
                "<PAD>",
            ]
            continue
        x_tmp.append(token)
        y_tmp.append(row.tag)
    return x, y
