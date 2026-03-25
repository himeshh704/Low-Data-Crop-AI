import streamlit as st
import requests
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg

st.set_page_config(page_title="Crop Stress AI", page_icon="🌿", layout="centered")

st.title("🌿 Few-Shot Crop Stress & Disease AI")
st.markdown("""
Welcome to the Low-Data Crop AI. This system uses **Prototypical Networks** and **Pseudo-Labeling** to classify crop imagery trained on extremely limited data.
""")

st.sidebar.header("System Status")
st.sidebar.info("Backend API is assumed to be running on localhost:8000")

uploaded_file = st.file_uploader("Upload a crop leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    if st.button("Detect Stress 🔍"):
        with st.spinner("Analyzing with Prototypical Centroids..."):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                response = requests.post("http://localhost:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    pred = data["prediction"]
                    conf = data["confidence"]
                    
                    if pred == "Healthy":
                        st.success(f"### Prediction: {pred} ({conf*100:.1f}%)")
                    elif pred == "Stressed":
                        st.warning(f"### Prediction: {pred} ({conf*100:.1f}%)")
                    else:
                        st.error(f"### Prediction: {pred} ({conf*100:.1f}%)")
                        
                    st.write("#### Confidence Breakdown:")
                    st.json(data["all_probabilities"])
                else:
                    st.error(f"Error from API: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection failed to backend API. Ensure `uvicorn app.api:app --reload` is running on port 8000!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
