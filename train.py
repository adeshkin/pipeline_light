import pytorch_lightning as pl
import wandb

from dataset_light import CustomDataModule
from model import LitModel
from callbacks import early_stop_callback, checkpoint_callback, wandb_logger, ImagePredictionLogger


def main():
    num_channels = 3
    input_size = (300, 224)
    num_classes = 10
    learning_rate = 2e-4

    height, width = input_size
    input_shape = (num_channels, height, width)

    image_dir = ""
    annotation_path = ""
    category = ""
    batch_size = 16

    dm = CustomDataModule(image_dir, annotation_path, category, batch_size, input_size)
    dm.setup()

    val_samples = next(iter(dm.val_dataloader()))

    model = LitModel(input_shape, num_classes, learning_rate)

    trainer = pl.Trainer(max_epochs=150,
                         progress_bar_refresh_rate=20,
                         gpus=1,
                         logger=wandb_logger,
                         callbacks=[early_stop_callback,
                                    ImagePredictionLogger(val_samples, batch_size)],
                         checkpoint_callback=checkpoint_callback)

    # Train the model
    trainer.fit(model, dm)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
