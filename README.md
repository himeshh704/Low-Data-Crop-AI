# Low-Data Crop Stress AI 🌿

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A production-ready, self-learning crop disease AI that trains with as few as **50 labelled images per class** — using Prototypical Few-Shot Networks + Pseudo-Labelling.

---

## 🧠 How It Works

| Stage | Method |
|---|---|
| Feature Extraction | ResNet50 (frozen early layers, fine-tune `layer4`) |
| Classification | Prototypical Networks (Euclidean distance to class centroids) |
| Semi-Supervised Expansion | Pseudo-labelling at ≥ 0.90 confidence with dynamic decay |
| API | FastAPI with `/predict`, `/health`, `/model-info` |
| UI | Next.js 14 + Tailwind + Framer Motion |

## Classes
- 🟢 **Healthy** (Tomato healthy leaves)
- 🟡 **Stressed** (Yellow Leaf Curl Virus)
- 🔴 **Diseased** (Early Blight)

---

## 🚀 Quick Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download real PlantVillage images (150 total, 50/class)
python download_real_data.py

# 3. Train the Prototypical Network
python train.py

# 4. (Optional) Run pseudo-label self-training loop
python self_train.py
```

## 🌐 Launch the System

**Backend (FastAPI — in Terminal 1):**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

**Frontend (Next.js — in Terminal 2):**
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:3000` 🎉

## 🐳 Docker
```bash
docker build -t crop-ai .
docker run -p 8501:8501 -p 8000:8000 crop-ai
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Upload image → get prediction + confidence |
| `/health` | GET | Server health, uptime, model status |
| `/model-info` | GET | Backbone info, algorithm details |
| `/docs` | GET | Interactive Swagger documentation |

## 📁 Project Structure
```
low_data_crop_ai/
├── models/          # backbone.py, few_shot.py, self_learning.py
├── utils/           # data_loader, augmentations, metrics, logger
├── app/             # api.py (FastAPI), streamlit_app.py
├── frontend/        # Next.js 14 + Tailwind UI
├── train.py         # Prototypical training loop
├── self_train.py    # Pseudo-labelling iteration
├── download_real_data.py  # Auto-fetch PlantVillage images
└── config.py        # All hyperparams & paths
```

## 📄 License
MIT — Free to use, modify and distribute.
