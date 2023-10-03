from math import pi as π

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from quantum_rotor.flows.autoregressive import AutoregressiveSigmoidFlow
from quantum_rotor.action import pbc_action

N_LATTICE = 10
BETA = 0.01
N_BATCH = 5000
N_TRAIN = 1000

def uniform(*shape):
    return torch.empty(shape).uniform_(0, 2 * π)

def train(flow):

    optimizer = Adam(flow.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_TRAIN)

    with trange(N_TRAIN + 1, desc="Training") as pbar:
        for step in pbar:
            u = uniform(N_BATCH, N_LATTICE, 1)
            φ, ldj = flow(u)
    
            S_pushfwd = ldj
            S_target = pbc_action(φ, BETA)

            log_weights = S_pushfwd - S_target
            loss = log_weights.mean().negative()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                ess = torch.exp(
                    log_weights.logsumexp(0).mul(2) - log_weights.mul(2).logsumexp(0)
                )
                ess /= N_BATCH
                pbar.set_postfix({"loss": f"{loss.item():.5f}", "ess": f"{ess.item():.3f}"})



def main():

    flow = AutoregressiveSigmoidFlow(
        n_lattice=N_LATTICE,
        n_mixture=4,
        net_shape=[16],
        net_activation="Tanh",
        weighted=True,
    )
    u = uniform(N_BATCH, N_LATTICE, 1)
    φ, ldj = flow(u)

    train(flow)



if __name__ == "__main__":
    main()
