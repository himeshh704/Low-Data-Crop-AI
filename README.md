# Low-Data Crop AI Pipeline 🌿

A fully autonomous, production-ready AI system built for classifying extremely sparse agricultural data using Prototypical Networks (Transfer + Few-Shot) and Semi-Supervised Self-Learning (Pseudo-Labeling).

## 🚀 Quick Setup (Without Docker)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Synthetic Data (For Testing Without Large Downloads)**
   ```bash
   python setup_dataset.py
   ```
3. **Train the Prototypical Network**
   ```bash
   python train.py
   ```
4. **Trigger the Pseudo-Labeling Self-Learning Loop**
   ```bash
   python self_train.py
   ```

## 🌐 Running the AI Services (Inference)

### FastAPI Backend
Provides a lightning-fast `/predict` endpoint computing distances against learned class prototypes.
```bash
uvicorn app.api:app --reload
```

### Streamlit UI
A beautiful frontend interface predicting crop damage interactively. (Ensure FastAPI is running simultaneously).
```bash
streamlit run app/streamlit_app.py
```

## 🐳 Docker Deployment
To entirely containerize and run both the FastAPI and Streamlit interfaces interactively:
```bash
docker build -t crop-ai .
docker run -p 8501:8501 -p 8000:8000 crop-ai
```
> Go to `http://localhost:8501` to view your deployed system!
