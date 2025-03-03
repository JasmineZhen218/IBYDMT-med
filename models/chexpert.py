import torch.nn as nn
from typing import Optional
from ibydmt import Constants as c, VisionLanguageModel
from torchvision.transforms import Resize, Normalize, Compose, ToTensor, CenterCrop
from torchvision.models import resnet50

from ibydmt import register_model
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from pathlib import Path

RESIZE = 512
CENTER_CROP_SIZE = 512


@register_model(name="cxr-bert-specialized")
class CheXpertModel(VisionLanguageModel):
    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        self.image_encoder = ImageInferenceEngine(
            image_model=get_biovil_t_image_encoder(),
            transform=create_chest_xray_transform_for_inference(
                resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
            ),
        )
        self.text_encoder = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.device = device

    def encode_text(self, text):
        text_features = self.text_encoder.get_embeddings_from_prompt(text)
        return text_features

    def encode_image(self, image_path):
        image_features = self.image_encoder.get_projected_global_embedding(
            image_path=Path(image_path)
        )
        dim = image_features.shape[0]
        image_features = image_features.reshape(-1, dim)
        return image_features


@register_model(name="chexpert-resnet")
class CheXpertClassifier(VisionLanguageModel):
    preprocess = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.0], std=[0.5]),
        ]
    )

    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        super().__init__()
        self.resnet = resnet50()
        dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(dim, 1)
        self.to(device)

    def forward(self, x):
        return self.resnet(x)

    def encode_text(self, text):
        raise NotImplementedError

    def encode_image(self, image_path):
        pass
