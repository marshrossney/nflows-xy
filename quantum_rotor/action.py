import torch
from torch.special import i0e

Tensor = torch.Tensor


def pbc_action(φ: Tensor, β: float):
    return -β * torch.cos(φ - φ.roll(-1, 1)).sum(dim=1)


def obc_action(φ: Tensor, β: float):
    return -β * torch.cos(φ[:, 1:] - φ[:, :-1]).sum(dim=1)
