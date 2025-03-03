import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Union
import requests
from torchvision.datasets.utils import check_integrity
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
import matplotlib.pyplot as plt
from skimage import io
import os
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

RESIZE = 512
CENTER_CROP_SIZE = 512

root = "/cis/home/zwang/IBYDMT-med/data"
Predictions = []
Labels = []
True_Positives = []
True_Negatives = []
False_Positives = []
False_Negatives = []
train_meta = os.path.join(root, "CheXpert-v1.0-small/train.csv")
train_meta = pd.read_csv(train_meta)
# img_txt_inference = _get_vlp_inference_engine()
_image_inference = ImageInferenceEngine(
        image_model=get_biovil_t_image_encoder(),
        transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE),
    )
_text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

for i in tqdm(range(10000)):
    image_path = os.path.join(root, train_meta['Path'][i])
    # get projected global embedding
    image_features = _image_inference.get_projected_global_embedding(image_path=Path(image_path)).reshape(1, -1)
    normal_text_features = _text_inference.get_embeddings_from_prompt("A chest X-ray without lung opacity")
    abnormal_text_features = _text_inference.get_embeddings_from_prompt("A chest X-ray with lung opacity")

    normal_score = cosine_similarity(image_features, normal_text_features)[0,0]
    abnormal_score = cosine_similarity(image_features, abnormal_text_features)[0,0]
    if normal_score > abnormal_score:
        prediction = 'Normal'
    else:
        prediction = 'Abnormal'
    if train_meta['Lung Opacity'][i] == 1:
        label = 'Abnormal'
    else:
        label = 'Normal'
    Predictions.append(prediction)
    Labels.append(label)
    if prediction == 'Abnormal' and label == 'Abnormal':
        True_Positives.append(train_meta['Path'][i])
    if prediction == 'Normal' and label == 'Normal':
        True_Negatives.append(train_meta['Path'][i])
    if prediction == 'Abnormal' and label == 'Normal':
        False_Positives.append(train_meta['Path'][i])
    if prediction == 'Normal' and label == 'Abnormal':
        False_Negatives.append(train_meta['Path'][i])

# write True Positives to a file
with open("True_Positives_lung_opacity.txt", "w") as f:
    for item in True_Positives:
        f.write("%s\n" % item)
f.close()
with open("True_Negatives_lung_opacity.txt", "w") as f:
    for item in True_Negatives:
        f.write("%s\n" % item)
f.close()
with open("False_Positives_lung_opacity.txt", "w") as f:
    for item in False_Positives:
        f.write("%s\n" % item)
f.close()
with open("False_Negatives_lung_opacity.txt", "w") as f:
    for item in False_Negatives:
        f.write("%s\n" % item)
f.close()


# write quality metrics to a file
import numpy as np
acc = np.sum(np.array(Predictions) == np.array(Labels)) / len(Predictions)
num_normal = np.sum(np.array(Labels) == 'Normal')
num_abnormal = np.sum(np.array(Labels) == 'Abnormal')
with open("quality_metrics_lung_opacity.txt", "w") as f:
    f.write("num_normal: %d\n" % num_normal)
    f.write("num_abnormal: %d\n" % num_abnormal)
    f.write("Accuracy: %f\n" % acc)
    # acc by class
    acc_normal = np.sum(np.array(Predictions)[np.array(Labels) == 'Normal'] == 'Normal') / num_normal
    acc_abnormal = np.sum(np.array(Predictions)[np.array(Labels) == 'Abnormal'] == 'Abnormal') / num_abnormal
    f.write("Accuracy for normal: %f\n" % acc_normal)
    f.write("Accuracy for abnormal: %f\n" % acc_abnormal)
f.close()
