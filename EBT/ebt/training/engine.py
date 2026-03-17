import torch
import torch.nn as nn
from ebt.model import energy_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_correct_energies = []
    all_incorrect_energies = []
    all_energy_gaps = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        energies = model(images)

        loss = energy_loss(energies, targets)
        total_loss += loss.item() * images.size(0)

        preds = energies.argmin(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

        batch_size = energies.size(0)
        correct_e = energies[torch.arange(batch_size, device=device), targets]
        all_correct_energies.append(correct_e.cpu())

        mask = torch.ones_like(energies, dtype=torch.bool)
        mask[torch.arange(batch_size, device=device), targets] = False
        incorrect_e = energies[mask].view(batch_size, -1).mean(dim=-1)
        all_incorrect_energies.append(incorrect_e.cpu())

        sorted_e = energies.sort(dim=-1).values
        gaps = sorted_e[:, 1] - sorted_e[:, 0]
        all_energy_gaps.append(gaps.cpu())

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "mean_correct_energy": torch.cat(all_correct_energies).mean().item(),
        "mean_incorrect_energy": torch.cat(all_incorrect_energies).mean().item(),
        "mean_energy_gap": torch.cat(all_energy_gaps).mean().item(),
    }
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, grad_clip=1.0):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        energies = model(images)
        loss = energy_loss(energies, targets)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            running_loss = total_loss / total_samples
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}]  loss={running_loss:.4f}")

    return total_loss / total_samples
