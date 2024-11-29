import os
import json
import threading
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import img_to_array
from utils.data_preprocessing import prepare_data
from utils.image_processing import preprocess_image

MODEL_DIR = "models/"
DATA_DIR = "data/"
def predict_image(image_file, confidence_threshold=0.65):
    model_path = os.path.join(MODEL_DIR, "trained_model.h5")
    class_map_path = os.path.join(MODEL_DIR, "class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        raise ValueError("No trained model found! Please train the model first.")

    model = load_model(model_path)
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)

    image = preprocess_image(image_file)

    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    class_name = class_map.get(str(predicted_class_index), "Unknown Class")
    return class_name, confidence