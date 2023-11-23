import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def prepare_data(root_dir, batch_size=4, num_workers=2, shuffle=True):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = ImageFolder(root_dir, train_transform)
    testset = ImageFolder(root_dir, test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader, trainset.classes