import torch, open_clip
from PIL import Image
import numpy as np

# Speed hint on CPU: set threads before any compute (PyTorch docs)
torch.set_num_threads(max(1, torch.get_num_threads()))  # or set an explicit small number like 4

class CLIPFeaturizer:
    def __init__(self, name="ViT-B-32", pretrained="laion2b_s34b_b79k", image_size=224, device="cpu"):
        self.device = device
        self.model, _, self.preproc = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval().to(device)
        # enforce square
        self.preproc.transforms[0].size = image_size

    @torch.inference_mode()
    def encode(self, pil_images):
        if not isinstance(pil_images, list): pil_images = [pil_images]
        batch = torch.stack([self.preproc(im).to(self.device) for im in pil_images])
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()
