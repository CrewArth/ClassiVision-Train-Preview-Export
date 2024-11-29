import os
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir, batch_size):
    """
    Prepares data for training and validation by loading images and augmenting them.
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for training and validation
    Returns:
        Tuple: training_data, validation_data
    """
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
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    validation_data = datagen.flow_from_directory(
        data_dir, target_size=(128, 128), batch_size=batch_size, subset="validation"
    )

    return train_data, validation_data
