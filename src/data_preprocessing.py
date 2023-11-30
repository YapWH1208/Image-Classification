import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

def prepare_data(root_dir: str, batch_size: int = 4, num_workers: int = 2, shuffle: bool = True, validation_split: float = 0.2):
    """
    Prepares the data for training, validation, and testing

    Args:
    root_dir (str): Path to the root directory of the dataset
    batch_size (int, optional): Batch size. Defaults to 4.
    num_workers (int, optional): Number of workers. Defaults to 2.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    validation_split (float, optional): Fraction of the dataset to be used as validation set. Defaults to 0.2.

    Returns:
    trainloader (DataLoader): DataLoader for training data
    valloader (DataLoader): DataLoader for validation data
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

    # Calculate the sizes of training, validation, and test sets
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training, validation, and test sets
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, dataset.classes
