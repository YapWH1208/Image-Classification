import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, batch_size=4, num_workers=2, shuffle = True):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.file_paths, self.labels = self._load_dataset()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_dataset(self):
        file_paths = []
        labels = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                file_paths.append(file_path)
                labels.append(self.class_to_idx[cls])
        return file_paths, labels