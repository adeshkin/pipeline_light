import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import CustomDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, annotation_path, category, batch_size, num_classes):
        super().__init__()

        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.category = category
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((300, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                        shear=None, resample=False, fillcolor=(255, 255, 255)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((300, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.image_dir,
                                           self.annotation_path,
                                           self.category,
                                           self.data_transforms["train"])

        self.val_dataset = CustomDataset(self.image_dir,
                                         self.annotation_path,
                                         self.category,
                                         self.data_transforms["val"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)