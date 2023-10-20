from dataclasses import dataclass
from math import pi as π, sqrt
import logging

import torch
from tqdm import trange

from nflows_xy.flows import Flow
from nflows_xy.xy import Action
from nflows_xy.core import PullbackAction
from nflows_xy.utils import mod_2pi

Tensor = torch.Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HmcMetrics:
    acceptance: float
    zero_stat: float
    err_zero_stat: float


def leapfrog(
    φ0: Tensor,
    p0: Tensor,
    action: Action,
    step_size: float,
    traj_length: float,
):
    assert p0.dtype == φ0.dtype

    n_steps = max(1, round(traj_length / abs(step_size)))

    φ = φ0.clone()
    p = p0.clone()
    ε = step_size
    t = 0

    F = action.grad(φ).negative()

    for _ in range(n_steps):
        p = p + (ε / 2) * F

        φ = mod_2pi(φ + ε * p)

        F = action.grad(φ).negative()

        p = p + (ε / 2) * F

        t += ε

    return φ, p, t


@torch.no_grad()
def hmc(
    action: Action,
    n_replica: int,
    n_traj: int,
    step_size: float,
    n_therm: int = 0,
    traj_length: float = 1.0,
) -> tuple[Tensor, HmcMetrics]:
    """Run the Hybrid Monte Carlo algorithm.

    Args:
        action: Target action to be sampled from.
        n_replica: Number of replica simulations running in parallel.
        n_traj: Number of molecular dynamics trajectories.
        step_size: Molecular dynamics timestep.
        n_therm: Number of trajectories to discard as 'thermalisation'.
        traj_length: Total time for each molecular dynamics trajectory.

    Returns:
        tuple[Tensor, HmcMetrics]:
            A tuple containing the sample of field configurations and the
            HMC metrics.
    """
    sample = torch.empty((n_traj, n_replica, *action.lattice_shape, 1))
    φ0 = torch.empty((n_replica, *action.lattice_shape, 1)).uniform_(0, 2 * π)

    total_accepted = 0
    ΔH_list = []

    SAMPLING = False
    with trange(n_therm + n_traj, desc="Thermalising") as pbar:
        for step in pbar:
            if step == n_therm:
                SAMPLING = True
                pbar.set_description_str("Sampling")

            φ0 = φ0.clone()
            p0 = torch.empty_like(φ0).normal_()
            H0 = 0.5 * p0.pow(2).sum(dim=-2) + action(φ0)

            φT, pT, T = leapfrog(
                φ0,
                p0,
                action,
                step_size,
                traj_length,
            )

            HT = 0.5 * pT.pow(2).sum(dim=-2) + action(φT)
            ΔH = HT - H0

            accepted = torch.squeeze(
                torch.exp(-ΔH) > torch.rand_like(ΔH),
                dim=1,
            )

            φ0[accepted] = φT[accepted]

            if SAMPLING:
                sample[step - n_therm] = φ0
                total_accepted += sum(accepted).item()
                ΔH_list.append(ΔH)

    r_accept = total_accepted / (n_traj * n_replica)
    ΔH = torch.stack(ΔH_list)
    expΔH = torch.exp(-ΔH)
    mean, stderr = expΔH.mean().item(), expΔH.std().item() / sqrt(
        expΔH.numel()
    )

    logger.info(f"Acceptance rate: {r_accept:.5f}")
    logger.info(f"exp(-ΔH) - 1 = {(mean - 1):e} ± {stderr:e}")

    metrics = HmcMetrics(r_accept, mean - 1, stderr)

    return sample.transpose(0, 1), metrics


def fhmc(
    flow: Flow,
    target: Action,
    n_replica: int,
    n_traj: int,
    step_size: float,
    n_therm: int = 0,
    traj_length: float = 1.0,
) -> tuple[Tensor, HmcMetrics]:
    """
    Alias for hmc(PullbackAction(flow, target), **kwargs).
    """
    u, metrics = hmc(
        PullbackAction(flow, target),
        n_replica=n_replica,
        n_traj=n_traj,
        step_size=step_size,
        traj_length=traj_length,
    )
    φ, _ = flow(u.flatten(0, 1))
    φ = φ.unflatten(0, u.shape[:2])
    return φ, metrics
