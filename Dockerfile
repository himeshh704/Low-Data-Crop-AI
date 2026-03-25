FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install PyTorch CPU directly to keep image size reasonable
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8000

RUN echo "#!/bin/bash" > start.sh
RUN echo "uvicorn app.api:app --host 0.0.0.0 --port 8000 &" >> start.sh
RUN echo "streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0" >> start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]
