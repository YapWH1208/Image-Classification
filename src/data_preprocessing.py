import os
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class CustomImageDataset:
    def __init__(self, root_dir, split='train', transform=None, test_size=0.2, val_size=0.1, random_state=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

        if split == 'train':
            train_paths, test_paths = train_test_split(self.image_paths, test_size=test_size, random_state=random_state)
            train_paths, val_paths = train_test_split(train_paths, test_size=val_size, random_state=random_state)
            self.image_paths = train_paths if split == 'train' else val_paths
        elif split == 'val':
            _, self.image_paths = train_test_split(self.image_paths, test_size=val_size, random_state=random_state)
        elif split == 'test':
            _, self.image_paths = train_test_split(self.image_paths, test_size=test_size, random_state=random_state)
        else:
            raise ValueError(f"Unsupported split type: {self.split}")

    def get_classes(self):
        class_set = set()
        for img_path in self.image_paths:
            class_name = os.path.basename(os.path.dirname(img_path))
            class_set.add(class_name)
        return sorted(list(class_set))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed_img = self.apply_transform(img)
            return transformed_img

        return img

    def apply_transform(self, img):
        if self.split == 'train':
            return self.transform_train(img)
        elif self.split == 'val':
            return self.transform_val(img)
        elif self.split == 'test':
            return self.transform_test(img)
        else:
            raise ValueError(f"Unsupported split type: {self.split}")

    def transform_train(self, img):
        return self.transform(img)

    def transform_val(self, img):
        return self.transform(img)

    def transform_test(self, img):
        return self.transform(img)


def prepare_data(root_dir: str, batch_size: int = 128, num_workers: int = 12, shuffle: bool = True, validation_split: float = 0.2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create DataLoader for training, validation, and test sets
    train_dataset = CustomImageDataset(root_dir=root_dir, split='train', transform=train_transform)
    val_dataset = CustomImageDataset(root_dir=root_dir, split='val', transform=transform)
    test_dataset = CustomImageDataset(root_dir=root_dir, split='test', transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader, testloader, train_dataset.classes


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