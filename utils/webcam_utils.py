import cv2
import os
import streamlit as st
import time
from PIL import Image


def initialize_camera():
    """
    Initialize the webcam. Tries indices 0 to 4 to find a working camera.
    Returns the camera object or None if no camera is found.
    """
    for index in range(5):  # Try indices 0 to 4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            st.info(f"Camera initialized at index {index}")
            return cap
    return None


def capture_images_from_webcam(output_dir):
    """
    Captures images from the webcam every 2 seconds until the "Stop Capture" button is pressed.
    Saves captured images to the specified directory and displays the webcam feed in Streamlit.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    captured_images = []  # List to store paths of captured images

    # Initialize camera
    cap = initialize_camera()
    if not cap:
        st.error("No accessible camera found. Please check connections or permissions.")
        return []

    # Streamlit session state for stopping capture
    if "stop_capture" not in st.session_state:
        st.session_state["stop_capture"] = False

    # Stop capture button
    stop_capture = st.button("Stop Capture")
    if stop_capture:
        st.session_state["stop_capture"] = True

    st.info("Capturing images. Press 'Stop Capture' to stop.")

    # Variables for capturing images
    start_time = time.time()
    image_count = 0

    # Streamlit placeholder for webcam feed
    st_frame = st.empty()

    while not st.session_state["stop_capture"]:
        # Capture frame from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB for Streamlit compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, caption="Webcam Feed", use_column_width=True)

        # Save an image every 2 seconds
        if time.time() - start_time >= 2:
            image_path = os.path.join(output_dir, f"image_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            captured_images.append(image_path)
            image_count += 1
            start_time = time.time()

        time.sleep(0.1)  # Small delay to prevent excessive resource usage

    # Release webcam and update UI
    cap.release()
    st.info("Webcam capture stopped.")

    return captured_images


# Main Streamlit App
def main():
    st.title("Webcam Image Capture")
    st.write("This app captures images from your webcam every 2 seconds.")

    # Directory for saving captured images
    output_dir = st.text_input("Enter the directory to save captured images:", "captured_images")

    if st.button("Start Webcam Capture"):
        st.write("Initializing webcam...")
        captured_images = capture_images_from_webcam(output_dir)

        if captured_images:
            st.success(f"Captured {len(captured_images)} images.")
            st.write("Captured Images:")
            for img_path in captured_images:
                st.image(Image.open(img_path), caption=img_path, use_column_width=True)


if __name__ == "__main__":
    main()
