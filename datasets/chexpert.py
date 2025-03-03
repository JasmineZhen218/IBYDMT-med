import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Mapping, List

from ibydmt import (
    get_predictions,
    register_dataset,
    register_dataset_concept_trainer,
    Config,
    Constants as c,
)

rng = np.random.default_rng()


def _lung_opacity_label(row):
    if row["Lung Opacity"] == 1:
        return 1
    if any(
        row[["Edema", "Consolidation", "Pneumonia", "Atelectasis", "Lung Lesion"]] == 1
    ):
        return 1
    return 0


LUNG_OPACITY_TASK = {
    "label": _lung_opacity_label,
    "classes": {
        "normal": "A photo of a chest X-ray without lung opacity",
        "abnormal": "A photo of a chest X-ray with lung opacity",
    },
}


@register_dataset(name="chexpert")
class CheXpert(VisionDataset):
    def __init__(self, root, train=None, transform=None):
        super().__init__(root, transform=transform)
        # self.op = "all"
        self.op = "train" if train else "test"
        self.load_image = False

        task = LUNG_OPACITY_TASK
        label_fn = task["label"]
        self.classes = list(task["classes"].keys())
        self.prompts = list(task["classes"].values())

        data_dir = os.path.join(root, "CheXpert-v1.0-small")
        meta_path = os.path.join(data_dir, "train.csv")
        meta_df = pd.read_csv(meta_path)
        valid = meta_df["Lung Opacity"] != -1
        meta_df = meta_df[valid]
        # meta_df = pd.read_csv(meta_path)[:10000]

        train_n, test_n = 50000, 10000
        if not train:
            meta_df = meta_df[:test_n]
        else:
            meta_df = meta_df[test_n : test_n + train_n]

        paths = list(meta_df["Path"])
        labels = meta_df.apply(label_fn, axis=1)
        print(train, labels.sum() / len(labels))
        self.samples = list(zip(paths, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        path = os.path.join(self.root, path)
        if self.load_image:
            image = Image.open(path)
            image = self.transform(image)
            return image, label
        return os.path.join(self.root, path), label

    def validate_class_idx_for_testing(
        self, config: Config, class_idx: Mapping[str, List[int]], workdir=c.WORKDIR
    ):
        meta_path = os.path.join(self.root, "CheXpert-v1.0-small", "train.csv")
        prediction_df = get_predictions(config, workdir=workdir)

        output = prediction_df.values[:, 1:]
        prediction = np.argmax(output, axis=-1)
        return {
            class_name: np.intersect1d(np.where(prediction == idx), _class_idx).tolist()
            for idx, (class_name, _class_idx) in enumerate(class_idx.items())
            if idx == 1
        }


@register_dataset_concept_trainer(name="chexpert")
def train_dataset_concepts(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
    positive_concepts = [
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Lung Lesion",
    ]
    negative_concepts = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumothorax",
        "Support Devices",
        "Fracture",
    ]
    return positive_concepts + negative_concepts
