import torch
import torch.nn as nn
from torchvision import transforms
import timm

def get_model():
    """Bangun arsitektur saja — JANGAN mengunduh bobot atau memuat model.pth di sini."""
    # pretrained=False karena akan memuat bobot dari model.pth
    # num_classes=5 mengatur agar output layer (logits) berukuran [batch, 5]
    model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=5)
    return model

def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    # Transformasi ini KHUSUS untuk evaluasi/validasi.
    # Resolusi bawaan EfficientNet-B2 adalah 260x260.
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform