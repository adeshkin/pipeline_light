import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from dataset_light import CustomDataModule
from model import LitModel
from callbacks import ImagePredictionLogger


def main():
    num_channels = 3
    input_size = (300, 224)
    num_classes = 10
    learning_rate = 2e-4

    height, width = input_size
    input_shape = (num_channels, height, width)

    model = LitModel(input_shape, num_classes, learning_rate)

    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min')

    trainer = pl.Trainer(max_epochs=150,
                         progress_bar_refresh_rate=20,
                         gpus=1,
                         logger=wandb_logger,
                         callbacks=[early_stop_callback,
                                    ImagePredictionLogger(val_samples, 64)],
                         checkpoint_callback=checkpoint_callback)

    # Train the model
    trainer.fit(model, dm)


    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
