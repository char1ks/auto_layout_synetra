import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add dinov3 to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dinov3')))

from dinov3.hub.backbones import dinov3_vitb16

class DinoV3Encoder:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = dinov3_vitb16(pretrained=True).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image_path_or_pil):
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
            
        return features.cpu().numpy()
