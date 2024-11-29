import streamlit as st
import cv2
import os
import time
from PIL import Image

def capture_images_from_webcam(output_dir):
    """
    Captures images from the webcam every 2 seconds and saves them in the specified directory.
    Displays the images in the Streamlit app. Allows stopping with a button.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    st.info("Press 'Stop' to stop capturing images.")

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    # Add a static Stop button with a unique key
    stop_capture = st.button("Stop Capture", key="stop_capture_button")

    try:
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
            st_frame.image(image, caption="Webcam Feed", use_container_width=400)

            # Save an image every 2 seconds
            if time.time() - start_time >= 2:
                image_path = os.path.join(output_dir, f"image_{image_count}.jpg")
                cv2.imwrite(image_path, frame)
                image_count += 1
                start_time = time.time()

            # Check if stop button was clicked
            if stop_capture:
                st.write("Webcam capture stopped.")
                break

            time.sleep(0.1)  # Small delay to avoid overloading the app

    finally:
        # Release the webcam resource
        cap.release()
        st.info("Webcam capture stopped.")
