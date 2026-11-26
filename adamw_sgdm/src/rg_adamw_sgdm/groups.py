"""Explicit parameter selection for general models and the Shakespeare recipe."""

import re

from torch import nn


_SHAKESPEARE_WEIGHT = re.compile(
    r"transformer\.h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight"
)


def parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float = 0.01,
    selection: str = "matrices",
    reference_lr_ratio: float = 100.0,
) -> list[dict]:
    """Partition trainable parameters without duplicating tied weights.

    ``matrices`` grafts tensors with ndim >= 2, except embedding weights and
    anything tied to them. ``shakespeare`` grafts only the four named block
    projections in each nanoGPT block, exactly as the original experiment.
    All ndim >= 2 tensors receive the requested decay; vectors/scalars get
    zero decay. Set weight_decay=0 for the frozen Shakespeare recipe.

    Call this on the underlying model before wrapping it in DDP. For models
    with different naming or decay conventions, pass explicit optimizer
    parameter groups instead. Each returned group includes ``param_names``
    for inspection and checkpoint metadata.
    """
    if selection not in ("matrices", "shakespeare"):
        raise ValueError("selection must be 'matrices' or 'shakespeare'")

    embedding_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag))
    }
    grouped: dict[tuple[bool, bool], dict] = {}
    seen: set[int] = set()
    grafted = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if selection == "shakespeare":
            graft = bool(_SHAKESPEARE_WEIGHT.fullmatch(name)) and parameter.ndim == 2
        else:
            graft = parameter.ndim >= 2 and id(parameter) not in embedding_ids
        grafted += int(graft)
        decay = parameter.ndim >= 2
        key = (graft, decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "param_names": [],
                "graft": graft,
                "weight_decay": weight_decay if decay else 0.0,
                "reference_lr_ratio": reference_lr_ratio,
            }
        grouped[key]["params"].append(parameter)
        grouped[key]["param_names"].append(name)

    if not grouped:
        raise ValueError("model has no trainable parameters")
    if selection == "shakespeare" and not grafted:
        raise ValueError(
            "No Shakespeare block matrices matched. Pass the unwrapped nanoGPT "
            "model or use explicit parameter groups for different names."
        )
    return list(grouped.values())
