import torch


def energy_loss(energies, targets):
    """
    EBM negative log-likelihood loss for classification.

    L = E(x, y+) + log Σ_c exp(-E(x, c))

    Term 1: E(x, y+)         — push correct-class energy DOWN
    Term 2: log Σ exp(-E)     — push all energies UP (log-partition)

    Numerically identical to F.cross_entropy(logits=-energies, targets).
    """
    batch_size = energies.shape[0]
    # Term 1: energy of the correct class
    correct_energy = energies[torch.arange(batch_size, device=energies.device), targets]

    # Term 2: log-partition function = log Σ_c exp(-E(x,c))
    log_partition = torch.logsumexp(-energies, dim=-1)

    loss = correct_energy + log_partition
    return loss.mean()
