import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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

    image_dir = ""
    annotation_path = ""
    category = ""
    batch_size = 16

    dm = CustomDataModule(image_dir, annotation_path, category, batch_size, input_size)
    dm.setup()

    val_samples = next(iter(dm.val_dataloader()))

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=1.0,
        patience=5,
        verbose=False,
        mode='max'
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model/model-{epoch:02d}-{val_acc:.2f}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename=MODEL_CKPT,
        save_top_k=5,
        mode='max'
    )

    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

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

    run = wandb.init(project='wandb-lightning', job_type='producer')

    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir(MODEL_CKPT_PATH)

    run.log_artifact(artifact)
    run.join()


if __name__ == "__main__":
    main()
