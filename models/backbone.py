import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_backbone(pretrained=True, freeze_early=True):
    """Loads ResNet50. Freezes early layers to prevent catastrophic forgetting on sparse datasets."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    backbone = resnet50(weights=weights)
    
    # Remove the final classification layer (fc) to get raw embeddings
    embedding_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    if freeze_early:
        # Freeze everything except the final layer block (layer4)
        for name, param in backbone.named_parameters():
            if not "layer4" in name:
                param.requires_grad = False
                
    return backbone, embedding_dim
