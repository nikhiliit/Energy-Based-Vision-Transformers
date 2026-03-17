import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_dataloaders(data_dir="./data", img_size=224, batch_size=64, num_workers=4):
    """
    Load Oxford-IIIT Pets dataset with train/val/test splits.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform_train = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    full_trainval = datasets.OxfordIIITPet(
        root=data_dir, split="trainval",
        target_types="category",
        transform=transform_train, download=True,
    )
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="test",
        target_types="category",
        transform=transform_val, download=True,
    )

    n_total = len(full_trainval)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_trainval, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset_clean = datasets.OxfordIIITPet(
        root=data_dir, split="trainval",
        target_types="category",
        transform=transform_val, download=False,
    )
    val_indices = val_dataset.indices
    val_dataset_final = Subset(val_dataset_clean, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset_final, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    num_classes = len(full_trainval.classes)
    class_names = full_trainval.classes

    print(f"Dataset: Oxford-IIIT Pets — {num_classes} classes")
    print(f"  Train: {n_train}  Val: {n_val}  Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, num_classes, class_names
