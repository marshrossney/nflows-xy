from math import pi as π, sqrt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange
import pandas as pd
from jsonargparse import CLI

from quantum_rotor.flows.autoregressive import AutoregressiveSigmoidFlow
from quantum_rotor.flows.coupling import SigmoidCouplingFlow
from quantum_rotor.action import Action, ActionPBC, ActionOBC, PullbackAction
from quantum_rotor.utils import mod_2pi

Tensor = torch.Tensor


def train(
    flow: nn.Module,
    target: Action,
    n_lattice: int,
    n_steps: int,
    n_batch: int,
    metrics_interval: int = 10,
    device: str = "cpu",
):
    flow = flow.to(device)

    optimizer = Adam(flow.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
    metrics_list = []

    with trange(n_steps + 1, desc="Training") as pbar:
        for step in pbar:
            u = torch.empty((n_batch, n_lattice, 1), device=device).uniform_(0, 2 * π)
            φ, ldj = flow(u)

            S_pushfwd = ldj
            S_target = target(φ)

            log_weights = S_pushfwd - S_target
            loss = log_weights.mean().negative()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % metrics_interval == 0:
                ess = (
                    torch.exp(
                        log_weights.logsumexp(0).mul(2)
                        - log_weights.mul(2).logsumexp(0)
                    )
                    / n_batch
                )
                vlw = log_weights.var()
                metrics = {
                    "step": step,
                    "loss": f"{loss.item():.5f}",
                    "ess": f"{ess.item():.3f}",
                    "vlw": f"{vlw.item():.3f}",
                }
                metrics_list.append(metrics)
                pbar.set_postfix(metrics)

    flow = flow.to("cpu")
    metrics = pd.DataFrame(metrics_list)

    return flow, metrics


def leapfrog(
    φ0: Tensor,
    p0: Tensor,
    action: Action,
    step_size: float,
    traj_length: float = 1.0,
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

        # print(torch.squeeze(0.5 * p.pow(2).sum(1) + action(φ), dim=1))

    return φ, p, t


def hmc(
    action: Action,
    n_lattice: int,
    n_replica: int,
    n_traj: int,
    n_therm: int,
    step_size: float,
    traj_length: float = 1.0,
):
    torch.set_default_dtype(torch.double)

    if isinstance(action, nn.Module):
        action = action.to(torch.double)

    φ0 = torch.empty((n_replica, n_lattice, 1), dtype=torch.double).uniform_(0, 2 * π)
    sample = torch.empty((n_traj, n_replica, n_lattice, 1), dtype=torch.double)
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
            H0 = 0.5 * p0.pow(2).sum(dim=1) + action(φ0)

            φT, pT, T = leapfrog(
                φ0,
                p0,
                action,
                step_size,
                traj_length,
            )

            HT = 0.5 * pT.pow(2).sum(dim=1) + action(φT)
            ΔH = HT - H0

            accepted = torch.squeeze(
                torch.exp(-ΔH) > torch.rand_like(ΔH),
                dim=1,
            )

            φ0[accepted] = φT[accepted]

            if SAMPLING:
                sample[step - n_therm] = φ0
                total_accepted += accepted.sum()
                ΔH_list.append(ΔH)

    r_accept = total_accepted / (n_traj * n_replica)
    print(f"Acceptance rate: {r_accept:.5f}")

    ΔH = torch.stack(ΔH_list)
    expΔH = torch.exp(-ΔH)
    mean, stderr = expΔH.mean().item(), expΔH.std().item() / sqrt(expΔH.numel())
    print(f"exp(-ΔH) - 1 = {(mean - 1):e} ± {stderr:e}")

    return sample


def test_hmc(
    n_lattice: int,
    beta: float,
):
    action = ActionPBC(beta)
    sample = hmc(
        action,
        n_lattice=n_lattice,
        n_replica=64,
        n_traj=1000,
        n_therm=100,
        step_size=0.1,
    )
    return sample


def auto(
    n_lattice: int,
    beta: float,
):
    flow = AutoregressiveSigmoidFlow(
        n_lattice=n_lattice,
        n_mixture=6,
        net_shape=[16],
        net_activation="Tanh",
        weighted=True,
    )
    action = ActionPBC(beta)

    trained_model, metrics = train(
        flow,
        action,
        n_lattice,
        n_steps=1000,
        n_batch=1024,
        device="cuda",
    )
    print(metrics)

    action = PullbackAction(action, trained_model)

    sample = hmc(
        action,
        n_lattice=n_lattice,
        n_replica=64,
        n_traj=100,
        n_therm=10,
        step_size=0.1,
    )


def coupling():
    flow = SigmoidCouplingFlow(
        n_blocks=1,
        n_lattice=N_LATTICE,
        n_mixture=6,
        net_shape=[16],
        net_activation="Tanh",
        weighted=True,
    )
    u = uniform(N_BATCH, N_LATTICE, 1)
    φ, ldj = flow(u)

    train(flow)


if __name__ == "__main__":
    # CLI(test_hmc)
    CLI(auto)
    # coupling()
