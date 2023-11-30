import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def prepare_data(root_dir:str, batch_size:int=4, num_workers:int=2, shuffle:bool=True):
    """
    Prepares the data for training and testing

    Args:
    root_dir (str): Path to the root directory of the dataset
    batch_size (int, optional): Batch size. Defaults to 4.
    num_workers (int, optional): Number of workers. Defaults to 2.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
    trainloader (DataLoader): DataLoader for training data
    testloader (DataLoader): DataLoader for testing data
    classes (list): List of classes in the dataset
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = ImageFolder(root_dir, transform)
    testset = ImageFolder(root_dir, transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader, trainset.classes