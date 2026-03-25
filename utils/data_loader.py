import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import cfg

logger = get_logger(__name__)

class CropDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(cfg.CLASSES)}
        
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist. Dataset will be empty.")
            return
            
        for cls_name in cfg.CLASSES:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.exists(cls_dir):
                logger.warning(f"Class directory {cls_dir} missing.")
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        logger.info(f"Loaded {len(self.image_paths)} images from {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a zero tensor if image is corrupt to prevent crashing
            image = torch.zeros((3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            return image, label, img_path

def create_dataloader(data_dir, transform, batch_size=cfg.BATCH_SIZE, shuffle=True):
    dataset = CropDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
