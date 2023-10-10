import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from nflows_xy.core import FlowBasedSampler

Tensor = torch.Tensor


@torch.enable_grad()
def train(
    model: FlowBasedSampler,
    n_steps: int,
    batch_size: int,
    init_lr: float = 1e-3,
    metrics_interval: int = 10,
):
    # Initialise lazy parameters
    _ = model(1)

    optimizer = Adam(model.parameters(), lr=init_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)

    metrics = []

    with trange(n_steps + 1, desc="Training") as pbar:
        for step in pbar:
            fields, actions = model(batch_size)

            log_weights = actions.pushforward - actions.target
            rev_kl = log_weights.mean().negative()
            loss = rev_kl  # TODO: include force minimisation in loss

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
                    / batch_size
                )
                vlw = log_weights.var()
                met = {
                    "step": step,
                    "loss": f"{loss.item():.5f}",
                    "ess": f"{ess.item():.3f}",
                    "vlw": f"{vlw.item():.3f}",
                }
                metrics.append(met)
                pbar.set_postfix(met)

    metrics = pd.DataFrame(metrics)

    return metrics


@torch.no_grad()
def test(
    model: FlowBasedSampler,
    batch_size: int = 1024,
    n_batches: int = 100,
):
    metrics = []

    with trange(n_batches, desc="Testing") as pbar:
        for _ in pbar:
            fields, actions = model(batch_size)

            log_weights = actions.pushforward - actions.target
            rev_kl = log_weights.mean().negative()
            loss = rev_kl

            ess = (
                torch.exp(
                    log_weights.logsumexp(0).mul(2)
                    - log_weights.mul(2).logsumexp(0)
                )
                / batch_size
            )
            vlw = log_weights.var()
            metrics.append(
                {
                    "loss": f"{loss.item():.5f}",
                    "ess": f"{ess.item():.3f}",
                    "vlw": f"{vlw.item():.3f}",
                }
            )

    metrics = pd.DataFrame(metrics)
    metrics = metrics.apply(pd.to_numeric)

    return metrics
