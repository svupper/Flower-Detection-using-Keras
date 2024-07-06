import pytorch_lightning as pl

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import PIL
import torch
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
# Before
# self.accuracy = pl.metrics.Accuracy()

# After
from torchmetrics import Accuracy


class Resnet50Model(pl.LightningModule):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.model = self.get_model()
        self.loss = nn.CrossEntropyLoss()
        # self.loss = criterion()
        # self.accuracy = Accuracy(num_classes=num_classes, task='MULTICLASS')
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def get_model(self):
        resnet = models.resnet50(pretrained=True)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, self.num_classes)
        return resnet

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        self.valid_acc(logits, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        return loss




    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        logits = self.forward(x)
        return logits



