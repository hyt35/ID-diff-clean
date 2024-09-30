from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from .sphere import KSphereDataset, KSphereScaledDataset
from .spaghetti import SpaghettiDataset
import numpy as np
import torch
import os

def get_dataloaders(args):
    dataset = args.dataset
    batch_size = args.batch_size
    if dataset == 'sphere':
        dataset = KSphereDataset(args)
    elif dataset == 'sphere_scaled':
        dataset = KSphereScaledDataset(args)
    elif dataset == 'spaghetti':
        dataset = SpaghettiDataset(args)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise NotImplementedError()
    # Determine sizes for train, val, and test sets
    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split dataset into train, val, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Example usage
# train_loader, val_loader, test_loader = get_dataloaders('sphere', batch_size=64)

def generate_and_save(args):
    dataset = args.dataset
    torch.manual_seed(0)
    if dataset == 'sphere':
        dataset = KSphereDataset(args)
    elif dataset == 'sphere_scaled':
        dataset = KSphereScaledDataset(args)
    elif dataset == 'spaghetti':
        dataset = SpaghettiDataset(args)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise NotImplementedError()

    os.makedirs(os.path.join(args.base_log_dir, args.experiment, "raw_data"), exist_ok=True)
    torch.save(dataset.data, os.path.join(args.base_log_dir, args.experiment, "points.pt"))
    return dataset.data

def get_saved_points(args):
    if not os.path.exists(os.path.join(args.base_log_dir, args.experiment, "points.pt")):
        points = generate_and_save(args)
    else:
        points = torch.load(os.path.join(args.base_log_dir, args.experiment, "points.pt"))
    return points