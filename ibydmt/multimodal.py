from abc import abstractmethod
from typing import Mapping, Optional

import clip
import open_clip
from transformers import (
    AlignModel,
    AlignProcessor,
    BlipForImageTextRetrieval,
    BlipProcessor,
    FlavaModel,
    FlavaProcessor,
)

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c


class VisionLanguageModel:
    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, image):
        pass


models: Mapping[str, VisionLanguageModel] = {}


def register_model(name):
    def register(cls: VisionLanguageModel):
        if name in models:
            raise ValueError(f"Model {name} is already registered")
        models[name] = cls

    return register


def get_model_name_and_backbone(config: Config):
    backbone = config.data.backbone.split(":")
    if len(backbone) == 1:
        backbone.append(None)
    return backbone


def get_model(config: Config, device=c.DEVICE) -> VisionLanguageModel:
    model_name, backbone = get_model_name_and_backbone(config)
    return models[model_name](backbone, device=device)


def get_text_encoder(config: Config, device=c.DEVICE):
    model = get_model(config, device=device)
    return model.encode_text


def get_image_encoder(config: Config, device=c.DEVICE):
    model = get_model(config, device=device)
    return model.encode_image


@register_model(name="clip")
class CLIPModel(VisionLanguageModel):
    def __init__(self, backbone: str, device=c.DEVICE):
        self.model, self.preprocess = clip.load(backbone, device=device)
        self.tokenize = clip.tokenize

        self.device = device

    def encode_text(self, text):
        text = self.tokenize(text).to(self.device)
        return self.model.encode_text(text)

    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image)


@register_model(name="open_clip")
class OpenClipModel(VisionLanguageModel):
    OPENCLIP_WEIGHTS = {
        "ViT-B-32": "laion2b_s34b_b79k",
        "ViT-L-14": "laion2b_s32b_b82k",
    }

    def __init__(self, backbone: str, device=c.DEVICE):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=self.OPENCLIP_WEIGHTS[backbone], device=device
        )
        self.tokenize = open_clip.get_tokenizer(backbone)

        self.device = device

    def encode_text(self, text):
        text = self.tokenize(text).to(self.device)
        return self.model.encode_text(text)

    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image)


@register_model(name="flava")
class FLAVAModel(VisionLanguageModel):
    HF_MODEL = "facebook/flava-full"

    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        if backbone is None:
            backbone = self.HF_MODEL

        self.model = FlavaModel.from_pretrained(backbone).to(device)
        self.processor = FlavaProcessor.from_pretrained(backbone)

        self.device = device

    def encode_text(self, text):
        text_inputs = self.processor(
            text=text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        return self.model.get_text_features(**text_inputs)[:, 0, :]

    def encode_image(self, image):
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        return self.model.get_image_features(**image_inputs)[:, 0, :]





@register_model(name="align")
class ALIGNModel(VisionLanguageModel):
    HF_MODEL = "kakaobrain/align-base"

    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        if backbone is None:
            backbone = self.HF_MODEL

        self.model = AlignModel.from_pretrained(backbone).to(device)
        self.processor = AlignProcessor.from_pretrained(backbone)

        self.device = device

    def encode_text(self, text):
        text_inputs = self.processor(
            text=text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        return self.model.get_text_features(**text_inputs)

    def encode_image(self, image):
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        return self.model.get_image_features(**image_inputs)


@register_model(name="blip")
class BLIPModel(VisionLanguageModel):
    HF_MODEL = "Salesforce/blip-itm-base-coco"

    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        if backbone is None:
            backbone = self.HF_MODEL

        self.model = BlipForImageTextRetrieval.from_pretrained(backbone).to(device)
        self.processor = BlipProcessor.from_pretrained(backbone)

        self.device = device

    def encode_text(self, text):
        text_inputs = self.processor(
            text=text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        question_embeds = self.model.text_encoder(**text_inputs)[0]
        return self.model.text_proj(question_embeds[:, 0, :])

    def encode_image(self, image):
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_embeds = self.model.vision_model(**image_inputs)[0]
        return self.model.vision_proj(image_embeds[:, 0, :])

#%%%%%%%%%%%%%%%%%%%%%%%%CLIP for Radiology%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from pathlib import Path
RESIZE = 512
CENTER_CROP_SIZE = 512
@register_model(name="cxr-bert-specialized")
class CXR_BERT_SpecializedModel(VisionLanguageModel):
    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        self.image_encoder = ImageInferenceEngine(
            image_model=get_biovil_t_image_encoder(),
            transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE),
        )
        self.text_encoder = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.device = device

    def encode_text(self, text):
        text_features = self.text_encoder.get_embeddings_from_prompt(text)
        return text_features

    def encode_image(self, image_path):
        image_features = self.image_encoder.get_projected_global_embedding(image_path=Path(image_path))
        dim = image_features.shape[0]
        image_features = image_features.reshape(-1, dim)
        return image_features
