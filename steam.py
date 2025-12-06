import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


IMG_SIZE = (256, 256) 
CLASS_NAMES = [
    "fish",
    "bass",
    "black_sea_sprat",
    "gilt_head_bream",
    "horse_mackerel",
    "red_mullet",
    "red_sea_bream",
    "sea_bass",
    "shrimp",
    "striped_red_mullet",
    "trout"
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Data science\Project5-imageclassification\CNN.h5")
    return model

model = load_model()

st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and the CNN model will predict the class.")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    
    img_resized = image.resize(IMG_SIZE)

    
    img_array = np.array(img_resized).astype("float32")

   
    img_array = img_array / 255.0

    
    img_batch = np.expand_dims(img_array, axis=0)

    
    preds = model.predict(img_batch)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    
    if 0 <= pred_index < len(CLASS_NAMES):
        pred_label = CLASS_NAMES[pred_index]
    else:
        pred_label = f"Class {pred_index}"

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2%}")
