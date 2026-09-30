import torch
import torch.nn as nn
from typing import Dict


def load_model_state(model: nn.Module, checkpoint_state: Dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    mismatched_keys = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            mismatched_keys.append(key)
            continue
        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    return list(incompatible.missing_keys), unexpected_keys, mismatched_keys
