import os
import cv2
from PIL import Image
import streamlit as st
import time

DATA_DIR = "data/"


def create_class_folder(class_name):
    class_folder = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)


def save_uploaded_images(files, class_name):
    create_class_folder(class_name)
    for file in files:
        image = Image.open(file)
        image.save(os.path.join(DATA_DIR, class_name, file.name))


def capture_webcam_images(class_name):
    st.info("Webcam started. Press 'Stop' when done.")

    # Access webcam
    cap = cv2.VideoCapture(0)

    # Ensure the class directory exists
    class_dir = f"{DATA_DIR}/{class_name}"
    os.makedirs(class_dir, exist_ok=True)

    # Create a container for the webcam feed
    placeholder = st.empty()

    # Place the "Stop" button outside the loop to avoid duplicate keys
    stop_capture = False

    # Add a separate stop button in a sidebar or a permanent UI section
    with st.sidebar:  # or another section
        stop_button = st.button("Stop Webcam")

    while not stop_capture:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the webcam feed in the placeholder
        placeholder.image(frame_rgb, caption="Webcam Feed", use_column_width=True)

        # Save the frame at 2-second intervals
        time.sleep(2)
        timestamp = int(time.time())
        cv2.imwrite(f"{class_dir}/{timestamp}.jpg", frame)

        # Check if the "Stop" button is clicked
        if stop_button:
            stop_capture = True

    cap.release()
    placeholder.empty()  # Clear the webcam feed
    st.success(f"Images captured and saved to {class_dir}.")
