from pathlib import Path
from torchvision import transforms
from transformers import (
    EfficientNetConfig,
    EfficientNetForImageClassification
)

class SampahEfficientNetB2(EfficientNetForImageClassification):
    def forward(self, x):
        return super().forward(pixel_values=x).logits

def get_model():
    config_path = Path(__file__).resolve().parent / "config.json"
    config = EfficientNetConfig.from_json_file(str(config_path))
    return SampahEfficientNetB2(config)

def get_transform():
    return transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163]
        )
    ])
