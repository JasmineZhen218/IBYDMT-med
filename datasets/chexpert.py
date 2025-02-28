import os

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset

from ibydmt.utils.concepts import register_class_concept_trainer
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import register_dataset

rng = np.random.default_rng()

import pandas as pd
@register_dataset(name="chexpert")
class CheXpert(VisionDataset):
    def __init__(self, root, train=None, transform=None):
        super().__init__(root, transform=transform)
        self.op = "train" if train else "test"
        self.classes = ['A normal chest X-ray', 'An abnormal chest X-ray']
        self.root = root
        image_root = os.path.join(root, "CheXpert-v1.0-small")
        if train:
            image_meta = pd.read_csv(
                os.path.join(image_root, "train_small.csv"))[:10]
            self.image_paths = list(image_meta['Path'])
            self.image_labels = list(image_meta['No Finding'])
        else:
            image_meta = pd.read_csv(
                os.path.join(image_root, "valid.csv"))
            self.image_paths = list(image_meta['Path'])
            self.image_labels = list(image_meta['No Finding'])
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        short_path = self.image_paths[idx]
        image_path = os.path.join(self.root, short_path)
        if self.image_labels[idx] == 1: # normal
            label = 0
        else:
            label = 1
        return image_path, label


@register_class_concept_trainer(name="chexpert")
def train_class_concepts(
    config: Config, concept_class_name: str, workdir=c.WORKDIR, device=c.DEVICE
):
    concepts = [
        "Enlarged Cardiom",
        "Cardiomegaly",
        "Lung Lesion",
        "Lung Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    return concepts
