import os
import sys
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from PIL import Image
import torch


class FashionMNISTEnhanced(FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class CIFAR10Enhanced(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed, dtype=torch.int64)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

def loading(dataset_name, data_path, device):
    if dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = FashionMNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = FashionMNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_data = CIFAR10Enhanced(data_path, transform=transform, download=True, device=device)
        test_data = CIFAR10Enhanced(data_path, transform=transform, train=False, device=device)
    else:
        raise Exception('Unknown dataset')

    return train_data, test_data
