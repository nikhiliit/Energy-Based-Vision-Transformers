import argparse
import os
import json
import torch
import numpy as np
from PIL import Image as PILImage
from sklearn.decomposition import PCA

from torch.optim.lr_scheduler import CosineAnnealingLR

from ebt.data import get_dataloaders
from ebt.model import get_model
from ebt.training import train_one_epoch, evaluate
from ebt.plotting import (
    plot_energy_landscape,
    plot_loss_curves,
    plot_energy_distributions,
    plot_attention_maps,
    plot_confusion_matrix,
    plot_ood_detection,
    plot_embeddings_tsne,
    plot_calibration,
    build_energy_surface,
    render_3d_frame
)


def run_training(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    train_loader, val_loader, _, num_classes, class_names = get_dataloaders(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    model = get_model(num_classes=num_classes, size=args.size, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "mean_correct_energy": [], "mean_incorrect_energy": [],
    }
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.grad_clip
        )
        metrics = evaluate(model, val_loader, device)

        for k, v in metrics.items():
            if k in history:
                history[k].append(v)
        history["train_loss"].append(loss)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "epoch": epoch,
                "args": vars(args)
            }, os.path.join(args.ckpt_dir, "ebt_best.pt"))

        print(f"Epoch {epoch}: Loss={loss:.4f} Val_Acc={metrics['accuracy']:.4f}")

    with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def run_analytics(args):
    device = torch.device(args.device if args.device != "auto" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(num_classes=ckpt["num_classes"], size=args.size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _, _, test_loader, _, class_names = get_dataloaders(batch_size=args.batch_size)
    
    print("Collecting model outputs...")
    all_energies, all_features, all_labels = [], [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            all_energies.append(model(imgs).cpu())
            all_features.append(model.get_features(imgs).cpu())
            all_labels.append(targets)
    
    energies = torch.cat(all_energies)
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    print("Generating IEEE plots...")
    plot_energy_landscape(model, features, labels, class_names, output_dir)
    plot_energy_distributions(energies, labels, output_dir)
    plot_confusion_matrix(energies, labels, class_names, output_dir)
    plot_ood_detection(model, energies, device, output_dir)
    plot_embeddings_tsne(features, labels, class_names, output_dir)
    plot_calibration(energies, labels, output_dir)
    
    history_path = os.path.join(os.path.dirname(args.checkpoint), "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curves(history, output_dir)
    
    print(f"All analytics saved to {output_dir}/")


def run_plot_3d(args):
    device = torch.device(args.device if args.device != "auto" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(num_classes=ckpt["num_classes"], size=args.size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _, _, test_loader, _, class_names = get_dataloaders(batch_size=args.batch_size)
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        feats = model.get_features(images.to(device)).cpu().numpy()
        energies = model(images.to(device)).cpu().numpy()

    num_classes = ckpt["num_classes"]
    if args.class_idx >= num_classes:
        raise ValueError(f"--class_idx {args.class_idx} out of range for {num_classes} classes")

    pca = PCA(n_components=2).fit(feats)
    feats_2d = pca.transform(feats)
    grid_bounds = (feats_2d[:, 0].min() - 1, feats_2d[:, 0].max() + 1,
                   feats_2d[:, 1].min() - 1, feats_2d[:, 1].max() + 1)

    xx, yy, eg = build_energy_surface(feats_2d, energies[:, args.class_idx], grid_bounds)

    kwargs = {"xx": xx, "yy": yy, "eg": eg, "azim": args.azim, "elev": args.elev}
    if args.style == "sample":
        kwargs.update({"pos": feats_2d[0], "color": "#FF2D55", "pil": PILImage.new("RGB", (224, 224), (200, 200, 200))})
    elif args.style == "dual":
        kwargs.update({"xx_a": xx, "yy_a": yy, "eg_a": eg, "xx_b": xx, "yy_b": yy, "eg_b": eg,
                       "pos_a": feats_2d[0], "pos_b": feats_2d[1]})

    frame = render_3d_frame(style=args.style, **kwargs)
    frame.save(args.output)
    print(f"Saved 3D plot ({args.style}) to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="EBT Modular Orchestrator")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--size", type=str, default="small")
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--grad_clip", type=float, default=1.0)
    train_p.add_argument("--ckpt_dir", type=str, default="checkpoints")

    # Analytics
    anal_p = subparsers.add_parser("analytics")
    anal_p.add_argument("--checkpoint", type=str, default="checkpoints/ebt_best.pt")
    anal_p.add_argument("--output_dir", type=str, default="assets/img")
    anal_p.add_argument("--size", type=str, default="small")

    # Plot 3D
    plot_p = subparsers.add_parser("plot_3d")
    plot_p.add_argument("--style", choices=["minimal", "sample", "dual"], default="minimal")
    plot_p.add_argument("--class_idx", type=int, default=0)
    plot_p.add_argument("--checkpoint", type=str, default="checkpoints/ebt_best.pt")
    plot_p.add_argument("--output", type=str, default="surface_3d.jpg")
    plot_p.add_argument("--size", type=str, default="small")
    plot_p.add_argument("--azim", type=float, default=-55)
    plot_p.add_argument("--elev", type=float, default=28)

    # Shared common args
    for p in [train_p, anal_p, plot_p]:
        p.add_argument("--data_dir", type=str, default="./data")
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--img_size", type=int, default=224)
        p.add_argument("--pretrained", action="store_true")
        p.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "analytics":
        run_analytics(args)
    elif args.mode == "plot_3d":
        run_plot_3d(args)

if __name__ == "__main__":
    main()
