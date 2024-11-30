import os
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from PIL import Image

# Directory to save captured images
OUTPUT_DIR = "captured_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.image_count = 0
        self.start_time = time.time()
        self.captured_images = []  # List to store captured image paths

    def recv(self, frame):
        # Convert the frame to OpenCV-compatible format
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB (Pillow-compatible format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save an image every 2 seconds
        if time.time() - self.start_time >= 2:
            image_path = os.path.join(OUTPUT_DIR, f"image_{self.image_count}.jpg")
            cv2.imwrite(image_path, img)  # Save the BGR image (OpenCV format)
            self.captured_images.append(image_path)
            self.image_count += 1
            self.start_time = time.time()

        # Return the RGB frame for live preview in Streamlit
        return img_rgb


def display_captured_images():
    """Display captured images in the sidebar."""
    st.sidebar.header("Captured Images")
    if os.listdir(OUTPUT_DIR):
        for image_file in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if image_file.endswith(".jpg"):
                st.sidebar.image(
                    Image.open(os.path.join(OUTPUT_DIR, image_file)),
                    caption=image_file,
                    use_column_width=True,
                )
    else:
        st.sidebar.info("No images captured yet.")


def main():
    st.title("Streamlit Webcam Capture App")
    st.info(
        "This app captures images from the webcam automatically every 2 seconds."
    )

    # Display captured images in the sidebar
    display_captured_images()

    # Start the webcam streamer
    ctx = webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},  # Video only
    )

    # Check if the video processor is active
    if ctx and ctx.video_processor:
        st.info("Webcam is active. Images are being captured every 2 seconds.")
    else:
        st.error("Webcam not accessible. Please ensure camera permissions are granted.")

    # Provide information about captured image storage
    st.sidebar.info("Images are saved in the 'captured_images' folder.")

    st.sidebar.warning(
        "Note: Ensure you have granted camera permissions and are using a browser with WebRTC support (e.g., Chrome, Firefox)."
    )


if __name__ == "__main__":
    main()
