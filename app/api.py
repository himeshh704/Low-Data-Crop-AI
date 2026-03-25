from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn.functional as F
from PIL import Image
import io
import sys
import os

# Ensure config models can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from models.few_shot import PrototypicalNetwork
from utils.augmentations import get_val_transforms
from utils.data_loader import create_dataloader
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Crop AI Few-Shot Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
prototypes = None
transform = get_val_transforms()

@app.on_event("startup")
def load_model():
    global model, prototypes
    model = PrototypicalNetwork().to(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_protonet.pth")
    
    if not os.path.exists(ckpt_path):
        logger.warning("No checkpoint found! API running on randomly initialized weights.")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        logger.info("Loaded best_protonet.pth")
        
    model.eval()
    
    # Pre-compute prototypes on startup
    loader = create_dataloader(cfg.FEW_SHOT_DIR, get_val_transforms(), batch_size=200, shuffle=False)
    if not loader:
        logger.warning("No support set found. Prototypes cannot be built.")
        return
        
    all_emb, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            all_emb.append(model(images.to(cfg.DEVICE)))
            all_labels.append(labels.to(cfg.DEVICE))
            
    prototypes = model.compute_prototypes(torch.cat(all_emb), torch.cat(all_labels))
    logger.info("Prototypes successfully pre-computed and cached.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, prototypes
    if not model or prototypes is None:
        raise HTTPException(status_code=503, detail="Model or Prototypes not initialized.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(cfg.DEVICE)
        
        with torch.no_grad():
            emb = model(tensor)
            logits = model.get_logits(emb, prototypes)
            probs = F.softmax(logits, dim=1)[0]
            
            conf, pred = torch.max(probs, dim=0)
            
        return {
            "prediction": cfg.CLASSES[pred.item()],
            "confidence": float(conf.item()),
            "all_probabilities": {cfg.CLASSES[i]: float(probs[i]) for i in range(cfg.NUM_CLASSES)}
        }
    except Exception as e:
        logger.error(f"Inference API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
