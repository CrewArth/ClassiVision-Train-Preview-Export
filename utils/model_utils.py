import os
import json
from PIL import Image, UnidentifiedImageError
import numpy as np
import streamlit as st
from keras._tf_keras.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras._tf_keras.keras.applications import MobileNet, ResNet50, VGG16
import matplotlib.pyplot as plt
from utils.loading_animation import  show_loading_animation

# Constants for directories
MODEL_DIR = "models/"
DATA_DIR = "data/"

# Global variable to store training history
training_history = None

def train_model(epochs, batch_size, learning_rate):
    global training_history
    if not os.path.exists(DATA_DIR):
        st.error("Dataset directory not found. Please add data and retry.")
        return

    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        classes=sorted(os.listdir(DATA_DIR))
    )

    validation_data = datagen.flow_from_directory(
        DATA_DIR, target_size=(128, 128), batch_size=batch_size, subset="validation"
    )

    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    # base_model.trainable = False
    # if base_model==choose_model['VGG16']:
    #     base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    # elif base_model==choose_model['ResNet50']:
    #     base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))


    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_data.num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    training_history = model.fit(train_data, validation_data=validation_data, epochs=epochs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "trained_model.h5"))
    st.success("Model trained successfully!")

    class_map_path = os.path.join(MODEL_DIR, "class_map.json")
    with open(class_map_path, 'w') as f:
        json.dump(train_data.class_indices, f)

    st.session_state['class_map'] = train_data.class_indices

def predict_image(image_file, confidence_threshold=0.65):
    model_path = os.path.join(MODEL_DIR, "trained_model.h5")
    class_map_path = os.path.join(MODEL_DIR, "class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        st.error("No trained model found! Please train the model first.")
        return

    model = load_model(model_path)
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)

    try:
        # Preprocess the uploaded image
        image = Image.open(image_file)
        image = image.convert("RGB")  # Ensure 3 channels
        image = image.resize((128, 128))
        image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]

        # Validate if the predicted class index exists in class_map
        class_indices = {v: k for k, v in st.session_state["class_map"].items()}

        if confidence < confidence_threshold:
            st.warning(f"Input image does not match any trained classes. (Confidence: {confidence:.2f})")
            # st.write("Prediction: Not Matched")
        else:
            class_name = class_indices.get(predicted_class_index, "Unknown Class")
            st.success(f"Prediction: {class_name}, Confidence: {confidence:.2f}")

        # Show the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=400, width=400)

    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid image.")
    except Exception as e:
        st.error(f"Error processing image: {e}")


def plot_training():
    global training_history
    if training_history is None:
        st.warning("No training history available!")
        return
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history["accuracy"], label="Training Accuracy")
    plt.plot(training_history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history["loss"], label="Training Loss")
    plt.plot(training_history.history["val_loss"], label="Validation Loss")
    plt.legend()
    st.pyplot(plt)

