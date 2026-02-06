
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Human Detection AI", page_icon="🧠")

st.title("🧠 AI PHÂN BIỆT NGƯỜI / KHÔNG PHẢI NGƯỜI")

model = tf.keras.models.load_model("human_classifier.h5")

img_file = st.file_uploader("📤 Upload ảnh", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file).convert("RGB").resize((96,96))
    st.image(img, width=300)

    x = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(x)[0][0]

    if pred > 0.5:
        st.success("✅ ĐÂY LÀ CON NGƯỜI")
    else:
        st.error("❌ KHÔNG PHẢI CON NGƯỜI")
