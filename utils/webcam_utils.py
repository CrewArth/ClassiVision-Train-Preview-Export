import cv2
import os
import streamlit as st
import time
from PIL import Image

# webcam_utils.py

def capture_images_from_webcam(output_dir):
    """
    Keeps the webcam open, captures frames automatically every 2 seconds,
    and saves them in the specified directory. Returns the list of captured images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    captured_images = []  # List to store captured images

    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return []

    st.info("Press 'Stop' to stop capturing images.")

    stop_capture = st.button("Stop Capture", key="stop_capture_button")

    image_count = 0
    start_time = time.time()

    # Streamlit placeholder for displaying frames
    st_frame = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image. Exiting...")
            break

        # Convert the frame to RGB (for Streamlit compatibility)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Display the frame in the Streamlit app
        st_frame.image(image, caption="Webcam Feed")

        # Save an image every 2 seconds
        if time.time() - start_time >= 2:
            image_path = os.path.join(output_dir, f"image_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            captured_images.append(image_path)  # Add image path to captured images list
            image_count += 1
            start_time = time.time()

        # Check if stop button was clicked
        if stop_capture:
            st.write("Webcam capture stopped.")
            break

        time.sleep(0.1)  # Small delay to avoid overloading the app

    # Release the webcam resource after stopping
    cap.release()
    st.info("Webcam capture stopped.")

    return captured_images  # Return the captured images
