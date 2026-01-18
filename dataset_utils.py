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

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=20, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=128, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)

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

def get_data_loaders(dataset):
    if dataset == "cifar100": return load_cifar100()
    if dataset == "mnist": return load_mnist()
    if dataset == "cifar10": return load_cifar10()