import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Class mapping ---
CLASS_NAMES = ['PNEUMONIA', 'NORMAL']  # 0=PNEUMONIA, 1=NORMAL
CLASS_LABELS = {'PNEUMONIA': 0, 'NORMAL': 1}

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('CNN.keras')

# --- Image preprocessing ---
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image.resize(target_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction function ---
def predict(image: Image.Image, model) -> tuple[dict, str, int]:
    img_preprocessed = preprocess_image(image)
    pred = model.predict(img_preprocessed, verbose=0)[0][0]
    class_confidences = {
        'PNEUMONIA': float(1 - pred),
        'NORMAL': float(pred)
    }
    predicted_class = max(class_confidences, key=class_confidences.get)
    predicted_label = CLASS_LABELS[predicted_class]
    return class_confidences, predicted_class, predicted_label

# --- Streamlit App UI ---
st.set_page_config(page_title="Chest X-Ray Classification", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🩺 Chest X-Ray Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an X-ray image to detect Pneumonia. Confidence values are shown below.</p>", unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            class_confidences, predicted_class, predicted_label = predict(image, model)
        
        # Highlight predicted class
        if predicted_class == "PNEUMONIA":
            st.markdown(f"<h2 style='color:#c62828;'>⚠️ Predicted Class: {predicted_class}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#2e7d32;'>✅ Predicted Class: {predicted_class}</h2>", unsafe_allow_html=True)
        
        st.markdown("### Confidence Degree:")
        
        # Custom horizontal bars
        for cls, conf in class_confidences.items():
            bar_color = "#2e7d32" if cls == "NORMAL" else "#c62828"
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <span style="font-weight:bold;">{cls}: {conf*100:.1f}%</span>
                <div style="background-color:#ddd; border-radius:5px; width:100%; height:20px;">
                    <div style="background-color:{bar_color}; width:{conf*100}%; height:20px; border-radius:5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
