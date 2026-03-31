from typing import Any

import torch


def resolve_strength(value, batch_size, device):
    if isinstance(value, (list, tuple)):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32)
        if value.dim() == 0:
            value = value.expand(batch_size)
        return value.view(batch_size, 1, 1)
    if value is None:
        value = 1.0
    return torch.full((batch_size, 1, 1), float(value), device=device)


def lookup_condition_embedding(ids, table, strength, reference):
    if table is None or ids is None:
        return 0.0
    if isinstance(ids, (list, tuple)):
        ids = torch.tensor(ids, device=reference.device, dtype=torch.long)
    elif not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, device=reference.device, dtype=torch.long)
    else:
        ids = ids.to(device=reference.device, dtype=torch.long)
    if ids.dim() == 0:
        ids = ids.expand(reference.size(0))
    valid = ids >= 0
    if valid.sum() <= 0:
        return 0.0
    lookup_ids = ids.clamp_min(0)
    embed = table(lookup_ids)[:, None, :]
    embed = embed * valid[:, None, None].float()
    return embed * strength


def project_scalar_condition(value, projector, batch_size, device):
    if projector is None or value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    elif not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    else:
        value = value.to(device=device, dtype=torch.float32)
    if value.dim() == 0:
        value = value.expand(batch_size)
    elif value.dim() == 2 and value.size(-1) == 1:
        value = value.squeeze(-1)
    value = value.to(device=device, dtype=torch.float32).view(batch_size, 1, 1)
    return projector(value)


def summary_vector(value: Any):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dim() == 3:
        return value.squeeze(1)
    return value


def squeeze_prompt_vector(value: Any):
    return summary_vector(value)
