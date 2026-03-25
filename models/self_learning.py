import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.few_shot import PrototypicalNetwork
from config import cfg
from utils.logger import get_logger

logger = get_logger(__name__)

class PseudoLabeler:
    def __init__(self, model: PrototypicalNetwork, confidence_threshold=cfg.PSEUDO_LABEL_THRESHOLD):
        self.model = model
        self.threshold = confidence_threshold
        
    @torch.no_grad()
    def generate_pseudo_labels(self, unlabeled_loader, prototypes):
        """Iterates over unlabeled pool, generating hard labels for high-confidence predictions."""
        self.model.eval()
        pseudo_dataset = [] # List of (image_tensor, label_int)
        new_labels_count = {c: 0 for c in range(cfg.NUM_CLASSES)}
        
        for images, _, paths in unlabeled_loader:
            images = images.to(cfg.DEVICE)
            
            embeddings = self.model(images)
            logits = self.model.get_logits(embeddings, prototypes)
            probs = F.softmax(logits, dim=1)
            
            max_probs, preds = torch.max(probs, dim=1)
            
            for i in range(len(max_probs)):
                if max_probs[i].item() >= self.threshold:
                    pred_class = preds[i].item()
                    # Store on CPU to avoid massive VRAM bloat
                    pseudo_dataset.append((images[i].cpu(), pred_class))
                    new_labels_count[pred_class] += 1
                    
        total_new = sum(new_labels_count.values())
        logger.info(f"Generated {total_new} pseudo-labels at Threshold {self.threshold:.2f}")
        logger.info(f"Class breakdown: {new_labels_count}")
        
        # Optional: Decay threshold dynamically if we are starving for data
        if total_new < 5:
            self.threshold *= cfg.CONFIDENCE_DECAY
            logger.info(f"Too few pseudo-labels. Decaying threshold to {self.threshold:.3f} for next round.")
            
        return pseudo_dataset
