import os
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import cv2


# Directory to save captured images
OUTPUT_DIR = "captured_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.image_count = 0
        self.start_time = time.time()
        self.captured_images = []  # List to store captured image paths

    def recv(self, frame):
        # Convert the frame to OpenCV-compatible format
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Capture and save an image every 2 seconds
        if time.time() - self.start_time >= 2:
            image_path = os.path.join(OUTPUT_DIR, f"image_{self.image_count}.jpg")
            cv2.imwrite(image_path, img)
            self.captured_images.append(image_path)
            self.image_count += 1
            self.start_time = time.time()

        # Return the frame for live preview
        return img


def main():
    st.title("Streamlit Webcam Capture App")
    st.info("This app captures images from the webcam automatically every 2 seconds.")

    # Sidebar to display captured images
    st.sidebar.header("Captured Images")
    if os.listdir(OUTPUT_DIR):
        for image_file in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if image_file.endswith(".jpg"):
                st.sidebar.image(os.path.join(OUTPUT_DIR, image_file), use_column_width=True)

    # Start webcam streaming
    ctx = webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)

    if ctx and ctx.video_processor:
        st.info("Webcam is active. Images are being captured every 2 seconds.")

    st.sidebar.info("To stop the app, press Ctrl+C or close the browser tab.")


if __name__ == "__main__":
    main()
