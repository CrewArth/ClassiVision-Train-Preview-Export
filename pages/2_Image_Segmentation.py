import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="deeplabv3.tflite")
    interpreter.allocate_tensors()
    return interpreter


# Preprocess the image (resizes to model's input size)
def preprocess_image(image, input_shape):
    resized_image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(resized_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# Resize the segmentation map to fixed 400x400 square dimensions
def resize_to_fixed(segmentation_map, output_size=(400, 400)):
    segmentation_image = Image.fromarray(segmentation_map.astype(np.uint8))
    return segmentation_image.resize(output_size, resample=Image.NEAREST)


# Apply a colormap to the segmentation map
def apply_colormap(segmentation_map):
    colormap = plt.get_cmap('tab20')  # Bright, distinct colors
    norm = mcolors.Normalize(vmin=0, vmax=np.max(segmentation_map))
    colored_map = colormap(norm(segmentation_map))  # Apply colormap
    return (colored_map[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB


# Perform segmentation
def segment_image(image):
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    preprocessed_image = preprocess_image(image, input_shape)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess the output
    segmentation_map = np.squeeze(output_data)
    segmentation_map = np.argmax(segmentation_map, axis=-1).astype(np.uint8)
    resized_map = resize_to_fixed(segmentation_map)  # Resize to 400x400 square
    return resized_map


# Main Streamlit app
def image_segmentation_page():
    st.title("Image Segmentation")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB").resize((400, 400))  # Resize input image to 400x400

        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded Image (400x400)", use_column_width=False, width=400)

        if st.button("Perform Segmentation"):
            with st.spinner("Segmenting..."):
                segmentation_map = segment_image(image)
                segmented_image_colored = apply_colormap(np.array(segmentation_map))
                segmented_image = Image.fromarray(segmented_image_colored)

            st.subheader("Segmented Image")
            st.image(segmented_image, caption="Segmented Image (400x400)", use_column_width=False, width=400)


# Run the app
if __name__ == "__main__":
    image_segmentation_page()
