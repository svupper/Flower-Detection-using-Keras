import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import os
from pathlib import Path
import requests
import shutil

import torchvision.transforms as transforms

class FlowersDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 5
        self.image_cat = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    def prepare_data(self):
        # download, split, etc...
        flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        # Download and extract the dataset

        if not os.path.exists("flower_photos.tgz"):
            response = requests.get(flowers_url, stream=True)
            with open("flower_photos.tgz", "wb") as file:
                shutil.copyfileobj(response.raw, file)

        os.makedirs("flower_photos", exist_ok=True)
        shutil.unpack_archive("flower_photos.tgz", extract_dir="flower_photos")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            data_dir = "flower_photos"
            transform = transforms.Compose([
                transforms.Resize((180, 180)),
                transforms.ToTensor()
            ])

            dataset = ImageFolder(data_dir, transform=transform)
            num_samples = len(dataset)
            train_size = int(0.8 * num_samples)
            val_size = num_samples - train_size

            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

