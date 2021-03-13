from PIL import Image
import torch.utils.data as data
import os
import csv


class CustomDataset(data.Dataset):
    def __init__(self, image_dir, annotation_path, category, tranforms=None):
        self.image_dir = image_dir
        if not os.path.exists(image_dir):
            raise Exception("Path {} does not exist".format(image_dir))

        if not os.path.exists(annotation_path):
            raise Exception("Path {} does not exist".format(annotation_path))

        self.image_paths = list()
        self.labels = list()
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.image_paths.append(row['image_path'])
                self.labels.append(row[category])

        self.transforms = tranforms

    def get_image(self, item):
        path = self.image_paths[item]
        path = os.path.join(self.image_dir, path)
        image = Image.open(path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        return image

    def get_class(self, item):
        label = self.labels[item]

        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        return self.get_image(item), self.get_class(item)
