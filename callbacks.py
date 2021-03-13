from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb


early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=1.0,
        patience=5,
        verbose=False,
        mode='max'
    )

MODEL_CKPT = 'model/model-{epoch:02d}-{val_acc:.2f}'
checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename=MODEL_CKPT,
        save_top_k=5,
        mode='max'
)

PROJECT = 'wandb-lightning'
wandb_logger = WandbLogger(project=PROJECT, job_type='train')


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs[:self.num_samples],
                                               preds[:self.num_samples],
                                               val_labels[:self.num_samples])]
        })