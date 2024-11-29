import os
from PIL import Image
import streamlit as st

def save_uploaded_images(uploaded_files, class_name):
    class_path = os.path.join("data", class_name)
    os.makedirs(class_path, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(class_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success(f"Saved {len(uploaded_files)} images to class '{class_name}'")
