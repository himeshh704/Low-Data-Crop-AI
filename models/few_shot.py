import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone import get_backbone
from config import cfg

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.embedding_dim = get_backbone(pretrained=True, freeze_early=True)
        
    def forward(self, x):
        return self.backbone(x)

    def compute_prototypes(self, support_emb, support_labels):
        """Computes the centroid prototype for each class in the few-shot episode."""
        prototypes = []
        for c in range(cfg.NUM_CLASSES):
            # Select embeddings natively belonging to this class
            class_mask = (support_labels == c)
            if class_mask.sum() == 0:
                # Fallback if a class gets completely masked out during sampling
                prototypes.append(torch.zeros(self.embedding_dim, device=support_emb.device))
            else:
                class_emb = support_emb[class_mask]
                class_proto = class_emb.mean(dim=0)
                prototypes.append(class_proto)
                
        return torch.stack(prototypes) # [NUM_CLASSES, EMBEDDING_DIM]

    def compute_distances(self, query_emb, prototypes):
        """Computes Euclidean distance between query embeddings and class prototypes."""
        n_query = query_emb.size(0)
        n_classes = prototypes.size(0)
        
        # Expand tensors for broadcasted subtraction
        q_exp = query_emb.unsqueeze(1) # [Q, 1, D]
        p_exp = prototypes.unsqueeze(0) # [1, C, D]
        
        distances = torch.pow(q_exp - p_exp, 2).sum(dim=2) # [Q, C]
        return distances

    def get_logits(self, query_emb, prototypes):
        """Returns logits based on negative distances. Max val = closest prototype."""
        distances = self.compute_distances(query_emb, prototypes)
        return -distances
