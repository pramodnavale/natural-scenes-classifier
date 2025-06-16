import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set page title
st.set_page_config(page_title="Scene Classifier", layout="centered")

st.title("Natural Scenes Image Classifier")
st.write("Upload an image of a natural scene, and the model will classify it as **buildings**, **forest**, or **sea**.")

# Load the trained model
model = load_model(r"scene_classifier.keras")

# Define class names
class_names = ['buildings', 'forest', 'sea']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Update if your model used different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Show results
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}")
