import os
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def prepare_data(root_dir: str, batch_size: int = 128, num_workers: int = 12, shuffle: bool = True, validation_split: float = 0.2):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root_dir)

    # Calculate the sizes of training, validation, and test sets
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = transform
    test_dataset.dataset.transform = transform

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)

    return trainloader, valloader, testloader, dataset.classes


def prepare_test_data(root_dir: str, batch_size: int = 4, num_workers: int = 2):
    """
    Prepares the data for testing

    Args:
    root_dir (str): Path to the root directory of the dataset
    batch_size (int, optional): Batch size. Defaults to 4.
    num_workers (int, optional): Number of workers. Defaults to 2.

    Returns:
    testloader (DataLoader): DataLoader for testing data
    classes (list): List of classes in the dataset
    """

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    dataset = ImageFolder(root_dir, transform)

    # Create DataLoader for testing set
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return testloader, dataset.classes