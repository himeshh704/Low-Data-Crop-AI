import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import cfg
from models.few_shot import PrototypicalNetwork
from utils.data_loader import create_dataloader
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.logger import get_logger

logger = get_logger(__name__)

def train_prototypical_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        embeddings = model(images)
        prototypes = model.compute_prototypes(embeddings, labels)
        
        logits = model.get_logits(embeddings, prototypes)
        
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(dataloader), correct / total

def main():
    logger.info("Starting baseline ProtoNet training...")
    
    train_loader = create_dataloader(cfg.FEW_SHOT_DIR, get_train_transforms(), batch_size=32)
    if train_loader is None:
        logger.error("No training data found. Run python setup_dataset.py first.")
        return
        
    model = PrototypicalNetwork().to(cfg.DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss, acc = train_prototypical_epoch(model, train_loader, optimizer, cfg.DEVICE)
        logger.info(f"Epoch {epoch}/{cfg.EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_protonet.pth"))
            logger.info("Saved new best model.")

if __name__ == "__main__":
    main()
