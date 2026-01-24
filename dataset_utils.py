import os
import shutil
import urllib.request
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_class, root, ds_mean, ds_std):
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std),
        ])
    if dataset_class == datasets.CIFAR100:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std),
        ])
    else:
        train_transform = test_transform

    train_ds = dataset_class(root=root, train=True, download=True, transform=train_transform)
    test_ds = dataset_class(root=root, train=False, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=128, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return train_loader, test_loader

def load_mnist():
    mean, std = torch.tensor([0.1307]), torch.tensor([0.3081])
    return load_dataset(dataset_class=datasets.MNIST, root="./data/mnist", ds_mean=mean.tolist(), ds_std=std.tolist())

def load_cifar10():
    mean, std = torch.tensor([0.5071, 0.4866, 0.4409]) , torch.tensor([0.2673, 0.2564, 0.2762])
    return load_dataset(dataset_class=datasets.CIFAR10, root="./data/cifar10", ds_mean=mean.tolist(), ds_std=std.tolist())

def load_cifar100():
    mean, std = torch.tensor([0.4914, 0.4822, 0.4465]) , torch.tensor([0.2023, 0.1994, 0.2010])
    return load_dataset(dataset_class=datasets.CIFAR100, root="./data/cifar100", ds_mean=mean.tolist(), ds_std=std.tolist())

def _prepare_tinyimagenet(root):
    target_dir = os.path.join(root, "tinyimagenet")
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")

    if not os.path.exists(target_dir):
        os.makedirs(root, exist_ok=True)
        print("Downloading TinyImageNet...")
        urllib.request.urlretrieve(
            "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
            zip_path
        )

        print("Unzipping...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)

        os.rename(
            os.path.join(root, "tiny-imagenet-200"),
            target_dir
        )
        os.remove(zip_path)

    # ---- fix val folder ----
    val_dir = os.path.join(target_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(images_dir):
        print("Fixing TinyImageNet val split...")

        with open(ann_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            img, cls = line.split("\t")[:2]
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.move(
                os.path.join(images_dir, img),
                os.path.join(cls_dir, img)
            )

        shutil.rmtree(images_dir)
        print("Val split fixed.")

def load_tinyimagenet(root="./data/tinyimagenet"):
    _prepare_tinyimagenet("./data")

    TINY_MEAN = [0.4802, 0.4481, 0.3975]
    TINY_STD  = [0.2302, 0.2265, 0.2262]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    train_ds = datasets.ImageFolder(
        root=f"{root}/train",
        transform=train_transform
    )
    train_classes = train_ds.class_to_idx

    test_ds = datasets.ImageFolder(
        root=f"{root}/val",
        transform=test_transform
    )
    test_ds.targets = [
        train_classes[test_ds.classes[t]]
        for t in test_ds.targets
    ]
    test_ds.class_to_idx = train_classes

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return train_loader, test_loader

def get_data_loaders(dataset):
    if dataset == "cifar100": return load_cifar100()
    if dataset == "mnist": return load_mnist()
    if dataset == "cifar10": return load_cifar10()
    if dataset == "tinyimagenet": return load_tinyimagenet()