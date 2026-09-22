import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="MNIST Handwritten Digit Predictor",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Predictor")
st.write("อัปโหลดรูปภาพตัวเลขเขียนมือ (0-9) เพื่อให้ AI ทำนายผล")

# โหลดโมเดลที่บันทึกไว้
model_path = '67102010158_mnist_model.keras'

@st.cache_resource
def load_mnist_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

model = load_mnist_model(model_path)

if model is None:
    st.error(f
