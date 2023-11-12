from math import pi as π, sqrt
import logging

import pandas as pd
import torch
from tqdm import trange

from nflows_xy.flows import Flow
from nflows_xy.xy import Action
from nflows_xy.core import PullbackAction
from nflows_xy.utils import mod_2pi

Tensor = torch.Tensor
DataFrame = pd.DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def winding_transform(φ: Tensor, extent: int | None = None) -> Tensor:
    n_batch, L, _ = φ.shape
    lw = extent or L

    assert lw >= 2 and lw <= L

    # Generate random starting point and orientation
    start = torch.randint(0, L - 1, (n_batch,)).tolist()
    orientation = torch.where(
        torch.rand(n_batch) < 0.5,
        torch.ones(n_batch),
        -torch.ones(n_batch),
    ).tolist()

    δφ = torch.cat([torch.arange(lw) * (2 * π) / lw, torch.zeros(L - lw)])
    δφ = torch.stack(
        [pm * δφ.roll(-x0, 0) for pm, x0 in zip(orientation, start)]
    ).unsqueeze(-1)

    return mod_2pi(φ + δφ)


@torch.no_grad()
def hmc(
    action: Action,
    n_replica: int,
    n_traj: int,
    step_size: float,
    n_therm: int = 0,
    traj_length: float = 1.0,
    use_winding: bool = False,
    wind_extent: int | None = None,
) -> tuple[Tensor, DataFrame]:
    """Run the Hybrid Monte Carlo algorithm.

    Args:
        action: Target action to be sampled from.
        n_replica: Number of replica simulations running in parallel.
        n_traj: Number of molecular dynamics trajectories.
        step_size: Molecular dynamics timestep.
        n_therm: Number of trajectories to discard as 'thermalisation'.
        traj_length: Total time for each molecular dynamics trajectory.
        use_winding: Propose a winding transformation before each trajectory.
        wind_extent: Fixed spatial extent of winding transformations.

    Returns:
        tuple[Tensor, HmcMetrics]:
            A tuple containing the sample of field configurations and the
            HMC metrics.
    """

    sample = torch.empty((n_traj, n_replica, *action.lattice_shape, 1))
    φ0 = torch.empty((n_replica, *action.lattice_shape, 1)).uniform_(0, 2 * π)

    history = {"ΔH": [], "accept_traj": []}
    if use_winding:
        history |= {"ΔSw": [], "accept_wind": []}

    SAMPLING = False
    with trange(n_therm + n_traj, desc="Thermalising") as pbar:
        for step in pbar:
            if step == n_therm:
                SAMPLING = True
                pbar.set_description_str("Sampling")

            if use_winding:
                φ0w = winding_transform(φ0, wind_extent)
                ΔSw = action(φ0w) - action(φ0)
                ΔSw.squeeze_(dim=1)
                accept_wind = torch.exp(-ΔSw) > torch.rand_like(ΔSw)
                φ0 = torch.where(accept_wind.view(-1, 1, 1), φ0w, φ0)

                if SAMPLING:
                    history["ΔSw"].append(ΔSw)
                    history["accept_wind"].append(accept_wind)

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
            ΔH.squeeze_(dim=1)  # squeeze to 1d
            accept_traj = torch.exp(-ΔH) > torch.rand_like(ΔH)

            φ0 = torch.where(accept_traj.view(-1, 1, 1), φT, φ0)

            if SAMPLING:
                sample[step - n_therm] = φ0
                history["ΔH"].append(ΔH)
                history["accept_traj"].append(accept_traj)

    history = {key: torch.stack(val) for key, val in history.items()}

    # Log the mean and standard error of exp(-ΔH) - 1
    expΔH = torch.exp(-history["ΔH"])
    mean = expΔH.mean().item()
    stderr = expΔH.std().item() / sqrt(expΔH.numel())
    logger.info(f"exp(-ΔH) - 1 = {(mean - 1):e} ± {stderr:e}")

    # Log the acceptance rate for trajectories
    r_accept = torch.mean(history["accept_traj"].float())
    logger.info(f"Mean trajectory acceptance rate: {r_accept:.3f}")

    # Log the acceptance rate for winding transformations
    if use_winding:
        r_accept = torch.mean(history["accept_wind"].float())
        logger.info(f"Mean winding acceptance rate: {r_accept:.3f}")

    # Return Multi-index DataFrame
    history = pd.concat(
        {key: pd.DataFrame(val) for key, val in history.items()}
    )

    return sample.transpose(0, 1), history


def fhmc(
    flow: Flow,
    target: Action,
    n_replica: int,
    n_traj: int,
    step_size: float,
    n_therm: int = 0,
    traj_length: float = 1.0,
) -> tuple[Tensor, DataFrame]:
    """
    Alias for hmc(PullbackAction(flow, target), **kwargs).
    """
    u, history = hmc(
        PullbackAction(flow, target),
        n_replica=n_replica,
        n_traj=n_traj,
        step_size=step_size,
        traj_length=traj_length,
    )
    φ, _ = flow(u.flatten(0, 1))
    φ = φ.unflatten(0, u.shape[:2])
    return φ, history
