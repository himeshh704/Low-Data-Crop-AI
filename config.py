import os
import torch
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    # Paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    FEW_SHOT_DIR = os.path.join(DATA_DIR, "few_shot_subset")
    UNLABELED_DIR = os.path.join(DATA_DIR, "unlabeled_pool")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # Dataset Classes
    CLASSES = ["Healthy", "Stressed", "Diseased"]
    NUM_CLASSES = len(CLASSES)

    # Few-Shot Parameters
    SAMPLES_PER_CLASS_LABELED = 50   # Simulating Extreme Low-Data
    SAMPLES_PER_CLASS_UNLABELED = 200 # Unlabeled pool for pseudo-labeling
    
    # Prototypical Network Hyperparams
    N_WAY = 3
    K_SHOT = 5
    Q_QUERY = 15

    # Self-Training Parameters
    PSEUDO_LABEL_THRESHOLD = 0.90
    CONFIDENCE_DECAY = 0.99  # dynamic threshold decay

    # Training Hyperparams
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 2
    IMAGE_SIZE = 224

    def __post_init__(self):
        # Auto-create all critical directories on init
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.FEW_SHOT_DIR, exist_ok=True)
        os.makedirs(self.UNLABELED_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

# Singleton export
cfg = Config()
