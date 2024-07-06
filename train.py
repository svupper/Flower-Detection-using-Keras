import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import FlowersDataModule
from model import Resnet50Model


def main():
    flower_data = FlowersDataModule()
    resnet_model = Resnet50Model()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="resnet", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(resnet_model, flower_data)


if __name__ == "__main__":
    main()
