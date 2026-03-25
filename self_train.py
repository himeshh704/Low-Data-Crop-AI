import torch
import os
import shutil
from config import cfg
from models.few_shot import PrototypicalNetwork
from models.self_learning import PseudoLabeler
from utils.data_loader import create_dataloader
from utils.augmentations import get_val_transforms
from utils.logger import get_logger
from train import main as train_baseline

logger = get_logger(__name__)

def run_pseudo_labeling():
    logger.info("Initializing Pseudo-Labeling Self-Training Loop...")
    
    model = PrototypicalNetwork().to(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_protonet.pth")
    if not os.path.exists(ckpt_path):
        logger.error("No baseline checkpoint found. Run train.py first.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
    model.eval()
    
    labeled_loader = create_dataloader(cfg.FEW_SHOT_DIR, get_val_transforms(), batch_size=100, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in labeled_loader:
            images = images.to(cfg.DEVICE)
            emb = model(images)
            all_embeddings.append(emb)
            all_labels.append(labels)
            
    support_emb = torch.cat(all_embeddings).to(cfg.DEVICE)
    support_labels = torch.cat(all_labels).to(cfg.DEVICE)
    
    prototypes = model.compute_prototypes(support_emb, support_labels)
    
    unlabeled_loader = create_dataloader(cfg.UNLABELED_DIR, get_val_transforms(), batch_size=32, shuffle=False)
    if unlabeled_loader is None:
        logger.info("No more unlabeled data to process!")
        return
        
    labeler = PseudoLabeler(model)
    pseudo_dataset = labeler.generate_pseudo_labels(unlabeled_loader, prototypes)
    
    moved = 0
    for img_cpu, pred_class, img_path in pseudo_dataset:
        cls_name = cfg.CLASSES[pred_class]
        dest_dir = os.path.join(cfg.FEW_SHOT_DIR, cls_name)
        
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, f"pseudo_{filename}")
        
        try:
            shutil.move(img_path, dest_path)
            moved += 1
        except Exception as e:
            logger.error(f"Failed to move {img_path}: {e}")
            
    logger.info(f"Successfully injected {moved} pseudo-labeled images into the training set!")
    
    if moved > 0:
        logger.info("Retraining baseline with expanded dataset...")
        train_baseline()

if __name__ == "__main__":
    run_pseudo_labeling()
