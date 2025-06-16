import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")


# Set page title
st.set_page_config(page_title="Scene Classifier", layout="centered")

st.title("Natural Scenes Image Classifier")
st.write("Upload an image of a natural scene, and the model will classify it as **buildings**, **forest**, or **sea**.")

model_path = os.path.join(os.path.dirname(__file__), "scene_classifier.keras")
model = load_model(model_path)

# Define class names
class_names = ['buildings', 'forest', 'sea']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image")

    # Preprocess the image
    img = img.resize((150, 150))  # Update if your model used different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Show results
    st.success(f"**Prediction:** {predicted_class}")

    #Show class probabilities
    st.subheader("Class Probabilities:")
    for class_name, prob in zip(class_names, predictions[0]):
        st.progress(int(prob * 100), text=f"{class_name}: {int(prob * 100)}%")
