import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ibydmt import register_dataset, Constants as c


def get_vindr_attributes(data_dir):
    vindr_dir = os.path.join(data_dir, "VinDr")
    meta_path = os.path.join(vindr_dir, "train.csv")
    meta_df = pd.read_csv(meta_path)
    attributes = meta_df["class_name"].unique().tolist()

    image_id = []
    image_df = meta_df.groupby("image_id")
    semantics = np.zeros((len(image_df), len(attributes)))

    for name, group in image_df:
        image_id.append(name)
        for _, row in group.iterrows():
            semantics[image_id.index(name), attributes.index(row["class_name"])] = 1

    df = {"image_id": image_id}
    for i, attr in enumerate(attributes):
        df[attr] = semantics[:, i]
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(vindr_dir, "attributes.csv"), index=False)


@register_dataset(name="vindr")
class VinDr(Dataset):
    def __init__(self, root, train=None, transform=None):
        super().__init__()
        self.op = "all"
        attribute = get_vindr_attributes(root)
        raise NotImplementedError
