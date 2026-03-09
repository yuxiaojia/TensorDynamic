from __future__ import annotations

from typing import List

import torch.nn as nn


def list_leaf_scopes(model: nn.Module) -> List[str]:
    """
    List all leaf Conv/Linear layers by their `named_modules()` scope.
    Used as the sweep target list.
    """
    names: List[str] = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0 and isinstance(
            m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        ):
            names.append(name)
    return names


