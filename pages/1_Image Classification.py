import streamlit as st
import os
import shutil
import time
import zipfile
from utils.data_utils import save_uploaded_images
from utils.model_utils import plot_training, predict_image, train_model
from utils.webcam_utils import  capture_images_from_webcam


# Constants for directories
DATA_DIR = "data/"
MODEL_DIR = "models/"
model_file_name = "trained_model.h5"
icon_path = "https://cdn-icons-png.flaticon.com/512/2557/2557436.png"

# Set page configuration
st.set_page_config(page_title="ClassiVision - Realtime Train, Preview & Effortless Download",
                    page_icon=icon_path,
                   layout="wide",
                   initial_sidebar_state="auto")

with st.spinner('Page is Loading...'):
    time.sleep(1)

st.markdown(
    """
    <style>
        body {
            background-color: #dbdbdb; /* Light grey background */
        }
        h2{
        margin-top:80px;
        }
        header {
            width:auto;
            padding-top: -40px;
            font-size:44px;
            color:#ffffff;
            text-align: center;
            padding: 20px;
            margin-top: -80px;
        }
        .logo {
            height:20px;
            display: inline-block;
            vertical-align: left;
            align-items:left;
            margin-right: 10px;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
            color: #ffffff; /* Dark grey */
        }
        .divider {
            border: none;
            border-top: 2px solid #cccccc; /* Light grey divider */
            margin-bottom: 20px;
        }
            </style>
    """,
    unsafe_allow_html=True,
)



# Initialize session state
if 'class_map' not in st.session_state:
    st.session_state['class_map'] = {}

if 'classes' not in st.session_state:
    st.session_state['classes'] = []

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = None

if 'download_ready' not in st.session_state:
    st.session_state['download_ready'] = False

if "training_in_progress" not in st.session_state:
    st.session_state["training_in_progress"] = False



# Header with External SVG logo
st.markdown(
    """
    <header>
        <div class="logo">
            <img src="https://cdn-icons-png.flaticon.com/512/2557/2557436.png" width="50" height="50" />
        </div>
        <div class="title">ClassiVision - Live Train, Predict & Export</div>
    </header>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Step 1: Class Management
st.header("1. Add Classes")
class_name = st.text_input("Enter Class Name", help=None)
if st.button("Add Class +"):
    if class_name:
        if class_name not in st.session_state['classes']:
            st.session_state['classes'].append(class_name)
            class_path = os.path.join(DATA_DIR, class_name)
            os.makedirs(class_path, exist_ok=True)
            st.write(f"Class '{class_name}' added!")
        else:
            st.warning("Class already exists!")
    else:
        st.warning("Class name cannot be empty!")

# Dropdown for class selection
if st.session_state['classes']:
    if 'previous_class' not in st.session_state:
        st.session_state['previous_class'] = None  # Initialize the previous class tracker

    selected_class = st.selectbox("Select Class", st.session_state['classes'])

    # Clear the uploaded files when switching classes
    if st.session_state['previous_class'] != selected_class:
        st.session_state['uploaded_files'] = {}  # Clear uploaded files
        st.session_state['previous_class'] = selected_class  # Update the current class

    st.write(f"Adding data for class: {selected_class}")

    # Data input options
    data_option = st.radio("Add Data", ["Webcam", "Upload Images"])
    if data_option == "Webcam":
        if st.button("Start Webcam Capture"):
            if selected_class:
                class_dir = os.path.join(DATA_DIR, selected_class)
                capture_images_from_webcam(class_dir)
            else:
                st.error("Please select a class first.")
    elif data_option == "Upload Images":
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

        # Save uploaded files in session state for the current class
        if uploaded_files:
            st.session_state['uploaded_files'][selected_class] = uploaded_files


        # Button to save files to backend
        if st.button("Save Uploaded Images"):
            save_uploaded_images(st.session_state['uploaded_files'].get(selected_class, []), selected_class)
else:
    st.info("Please add classes first to proceed.")



# Step 3: Training
st.header("3. Train Model")
if st.session_state['classes']:
    epochs = st.slider("Epochs", 1, 100, 10)
    batch_size = st.selectbox("Batch Size", [8, 16, 32])
    learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01])
    # choose_model = st.selectbox("Choose Pretrained Model ***(MobileNet Default)***", ['MobileNet','VGG16', 'ResNet50'])

    if st.button("Train Model"):
        if len(os.listdir(DATA_DIR)) >= 2:  # Ensure at least 2 classes have data
            st.write("Model is training...")
            train_model(epochs, batch_size, learning_rate)  # Pass user inputs to the backend
            st.session_state['model_trained'] = True
            plot_training()  # Show training results

        else:
            st.error("Please add at least 2 classes with images.")
else:
    st.info("No classes available. Add at least 2 classes before training.")

if st.session_state['model_trained']:
    st.session_state['download_ready'] = True


# Step 4: Prediction
st.header("4. Prediction")
if st.session_state['model_trained']:
    prediction_option = st.radio("Choose Prediction Mode", ["Upload Image"])
    if prediction_option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            predict_image(uploaded_image)
else:
    st.info("Please train the model first.")

st.write("Model can make mistakes.")
# Main "Download Model" button
if st.button("Download Model"):
    st.session_state['download_model_clicked'] = True
    st.toast("Please wait for other options.", icon="🔔")
# State variable to track the "Download Model" button click
if 'download_model_clicked' not in st.session_state:
    st.session_state['download_model_clicked'] = False

# Show options if "Download Model" is clicked
if st.session_state['download_model_clicked']:
    # Check if model exists
    if os.path.isfile(os.path.join(MODEL_DIR, model_file_name)):
        # Download ZIP Button
        zip_path = "data_models.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(DATA_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=DATA_DIR)
                    zipf.write(file_path, arcname=os.path.join("data", arcname))
            for root, dirs, files in os.walk(MODEL_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=MODEL_DIR)
                    zipf.write(file_path, arcname=os.path.join("models", arcname))

        # Provide ZIP download button
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download ZIP",
                data=f,
                file_name="data_models.zip",
                mime="application/zip"
            )


        # Convert to TensorFlow Lite and provide TFLite download button
        import tensorflow as tf

        tflite_model_path = os.path.join(MODEL_DIR, "trained_model.tflite")
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_file_name))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save TFLite model
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        # Provide TFLite download button
        with open(tflite_model_path, "rb") as f:
            st.download_button(
                label="Download TFLite Model",
                data=f,
                file_name="trained_model.tflite",
                mime="application/octet-stream"
            )
    else:
        st.write("No Model Trained yet. Train the Model to Download.")

# Reset and Quit
st.header("Quit and Reset")
if st.button("Quit and Reset"):
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        st.success("All data and models have been reset.")
    else:
        st.write("There is nothing to Reset.")
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    else:
        st.write(" ")
    for key in list(st.session_state.keys()):
        del st.session_state[key]



