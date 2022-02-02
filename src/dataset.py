# -*- coding: utf-8 -*-
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def load_data(data_dir="/blue/fsaeed/paulinaacostacev/Peptide-Modification-Classifier/pickle_files"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset