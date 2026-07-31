import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel


class CLIPVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = CLIPVisionModel(CLIPVisionConfig())

    def forward(self, images):
        return self.vision_model(pixel_values=images)


class FrozenCLIPEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CLIPVision()
        self.vision.eval()
        for parameter in self.vision.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        return self.vision(images).last_hidden_state

    def encode(self, images):
        return self(images)
