import os
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import img_to_array
from utils.data_preprocessing import prepare_data
from utils.image_processing import preprocess_image

MODEL_DIR = "models/"
DATA_DIR = "data/"

def train_model(epochs, batch_size, learning_rate):
    if not os.path.exists(DATA_DIR):
        raise ValueError("Dataset directory not found. Please add data and retry.")

    train_data, validation_data = prepare_data(DATA_DIR, batch_size)

    base_model = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "trained_model.h5"))

    with open(os.path.join(MODEL_DIR, "class_map.json"), 'w') as f:
        json.dump(train_data.class_indices, f)

    st.session_state['class_map'] = train_data.class_indices
    return history