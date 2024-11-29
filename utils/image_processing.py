from PIL import Image
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array

def preprocess_image(image_file):
    """
    Preprocesses the image before passing it to the model for prediction.
    Args:
        image_file (UploadedFile): The uploaded image file
    Returns:
        np.array: Preprocessed image array ready for prediction
    """
    image = Image.open(image_file)
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((128, 128))  # Resize image to match the model input shape
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array
