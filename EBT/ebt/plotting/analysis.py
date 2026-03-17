import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import torch.nn.functional as F

CORRECT_COLOR = "#3C3489"
INCORRECT_COLOR = "#C04828"
ACCENT_COLOR = "#4E89AE"
GRAY_COLOR = "#888780"
IEEE_DPI = 300


def apply_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "savefig.dpi": IEEE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.tick_params(direction="in", length=2)


def plot_energy_landscape(model, features, labels, class_names, output_dir):
    apply_ieee_style()
    feats_np = features.numpy()
    labels_np = labels.numpy()
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats_np)

    unique_classes = np.unique(labels_np)
    select_classes = unique_classes[np.linspace(0, len(unique_classes) - 1, 5, dtype=int)]
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.8))

    for ax, cls_idx in zip(axes, select_classes):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        x_min, x_max = feats_2d[:, 0].min() - 1, feats_2d[:, 0].max() + 1
        y_min, y_max = feats_2d[:, 1].min() - 1, feats_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100),
        )
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_full = pca.inverse_transform(grid_2d)

        with torch.no_grad():
            W = model.energy_head.weight[cls_idx].cpu()
            b = model.energy_head.bias[cls_idx].cpu()
            logits_cls = torch.tensor(grid_full, dtype=torch.float32) @ W + b
            energy_grid = (-logits_cls.numpy()).reshape(xx.shape)

        ax.imshow(
            energy_grid,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower", cmap="magma_r", aspect="auto", alpha=0.9,
        )
        mask = labels_np == cls_idx
        ax.scatter(
            feats_2d[mask, 0], feats_2d[mask, 1],
            c="white", s=6, alpha=0.7, edgecolors=CORRECT_COLOR, linewidths=0.3,
        )
        ax.set_title(cls_name.replace("_", " "), fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        clean_axes(ax, grid=False)

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(output_dir, "energy_landscape.jpg"))
    plt.close()


def plot_loss_curves(history, output_dir):
    apply_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.2))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color=INCORRECT_COLOR, label="Train", alpha=0.4)
    ax1.plot(epochs, history["val_loss"], color=CORRECT_COLOR, label="Val")
    ax1.set_title("Training Loss")
    ax1.legend(frameon=False)
    clean_axes(ax1)

    ax2.plot(epochs, [a * 100 for a in history["val_accuracy"]], color=ACCENT_COLOR)
    ax2.set_title("Validation Accuracy")
    clean_axes(ax2)

    ax3.plot(epochs, history["mean_correct_energy"], color=CORRECT_COLOR, label="$E(y^+)$")
    ax3.plot(epochs, history["mean_incorrect_energy"], color=INCORRECT_COLOR, label="$\\bar{E}(y^-)$")
    ax3.set_title("Energy Gap")
    ax3.legend(frameon=False)
    clean_axes(ax3)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(output_dir, "loss_curves.jpg"))
    plt.close()


def plot_energy_distributions(energies, labels, output_dir):
    apply_ieee_style()
    batch_size, _ = energies.shape
    correct_energies = energies[torch.arange(batch_size), labels].numpy()
    mask = torch.ones_like(energies, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    incorrect_energies = energies[mask].view(batch_size, -1).mean(dim=-1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.2))
    bins = np.linspace(
        min(correct_energies.min(), incorrect_energies.min()),
        max(correct_energies.max(), incorrect_energies.max()),
        64,
    )

    ax1.hist(correct_energies, bins=bins, color=CORRECT_COLOR, alpha=0.4,
             label="Correct $E(y^+)$", density=True)
    ax1.hist(incorrect_energies, bins=bins, color=INCORRECT_COLOR, alpha=0.4,
             label="Incorrect $\\bar{E}(y^-)$", density=True)
    ax1.set_title("Energy Alignment")
    ax1.legend(frameon=False)
    clean_axes(ax1, grid=False)

    gap = incorrect_energies - correct_energies
    ax2.hist(gap, bins=50, color=ACCENT_COLOR, alpha=0.3, density=True)
    ax2.axvline(x=0, color=GRAY_COLOR, linestyle="--")
    ax2.set_title("Separation Margin")
    clean_axes(ax2, grid=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(output_dir, "energy_distributions.jpg"))
    plt.close()


def plot_attention_maps(model, raw_dataset, test_dataset, device, output_dir, n_samples=8):
    apply_ieee_style()
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    fig, axes = plt.subplots(2, n_samples, figsize=(7.16, 2.5))

    for col, idx in enumerate(indices):
        raw_img, _ = raw_dataset[idx]
        norm_img, _ = test_dataset[idx]
        raw_np = np.array(raw_img)
        h, w = raw_np.shape[:2]

        with torch.no_grad():
            attn_maps = model.get_attention_maps(norm_img.unsqueeze(0).to(device))
            # last layer, average across heads, cls-token row
            attn = attn_maps[-1][0]
            cls_attn = attn[:, 0, 1:].mean(dim=0)
            grid_size = int(cls_attn.shape[0] ** 0.5)
            attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
            t = F.interpolate(
                torch.tensor(attn_map)[None, None], size=(h, w), mode="bilinear"
            )[0, 0].numpy()
            vis = np.clip(
                (t - np.percentile(t, 5)) / (np.percentile(t, 99) - np.percentile(t, 5) + 1e-8),
                0, 1,
            )

        axes[0, col].imshow(raw_np)
        axes[0, col].axis("off")
        axes[1, col].imshow(raw_np)
        axes[1, col].imshow(vis, cmap="inferno", alpha=0.55)
        axes[1, col].axis("off")

    plt.subplots_adjust(wspace=0.03, hspace=0.05)
    plt.savefig(os.path.join(output_dir, "attention_maps.jpg"))
    plt.close()


def plot_confusion_matrix(energies, labels, class_names, output_dir):
    apply_ieee_style()
    preds = energies.argmin(dim=-1).numpy()
    labels_np = labels.numpy()
    cm = confusion_matrix(labels_np, preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.8)
    ax.set_title("Breed Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))
    plt.close()


def plot_ood_detection(model, energies, device, output_dir, chunk_size=64):
    apply_ieee_style()
    in_dist = -torch.logsumexp(-energies, dim=-1).numpy()

    n_samples = min(500, len(energies))
    noise_scores = []
    for start in range(0, n_samples, chunk_size):
        batch = min(chunk_size, n_samples - start)
        noise_imgs = torch.randn(batch, 3, 224, 224, device=device)
        with torch.no_grad():
            noise_e = model(noise_imgs)
        noise_scores.append(-torch.logsumexp(-noise_e, dim=-1).cpu())
    noise_scores = torch.cat(noise_scores).numpy()

    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    bins = np.linspace(
        min(in_dist.min(), noise_scores.min()),
        max(in_dist.max(), noise_scores.max()),
        52,
    )
    ax.hist(in_dist, bins=bins, color=CORRECT_COLOR, alpha=0.4, label="In-dist (Pets)", density=True)
    ax.hist(noise_scores, bins=bins, color=INCORRECT_COLOR, alpha=0.4, label="OOD (Noise)", density=True)
    ax.set_title("OOD Detection")
    ax.legend(frameon=False)
    clean_axes(ax, grid=False)

    plt.savefig(os.path.join(output_dir, "ood_detection.jpg"))
    plt.close()


def plot_embeddings_tsne(features, labels, class_names, output_dir):
    apply_ieee_style()
    feats_np = features.numpy()
    if feats_np.shape[1] > 50:
        feats_np = PCA(n_components=50).fit_transform(feats_np)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    feats_2d = tsne.fit_transform(feats_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab20", len(class_names))
    for i in range(len(class_names)):
        mask = labels.numpy() == i
        if mask.any():
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=[cmap(i % 20)], s=5, alpha=0.6)

    ax.set_title("Geometric Separation of Breed Latents")
    clean_axes(ax)
    plt.savefig(os.path.join(output_dir, "embeddings_tsne.jpg"))
    plt.close()


def plot_calibration(energies, labels, output_dir):
    apply_ieee_style()
    preds = energies.argmin(dim=-1)
    correct = (preds == labels).numpy().astype(float)
    sorted_e = energies.sort(dim=-1).values
    gap = (sorted_e[:, 1] - sorted_e[:, 0]).numpy()

    n_bins = 10
    edges = np.percentile(gap, np.linspace(0, 100, n_bins + 1))
    accs = []
    for i in range(n_bins):
        mask = (gap >= edges[i]) & (gap <= edges[i + 1])
        accs.append(correct[mask].mean() if mask.any() else 0)

    plt.figure(figsize=(7.16, 2.2))
    plt.bar(range(n_bins), [a * 100 for a in accs], color=CORRECT_COLOR, alpha=0.7)
    plt.title("Confidence Calibration")
    clean_axes(plt.gca())
    plt.savefig(os.path.join(output_dir, "energy_gap_calibration.jpg"))
    plt.close()
