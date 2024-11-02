# Import necessary libraries
import numpy as np
import yaml
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image,ImageFile
import logging
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.get_logger().setLevel(logging.ERROR)

# Optionally handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagePredictor:
    def __init__(self, model_path, resize_size, targets, pre_processing_function=preprocess_input):
        """
        Initialize the ImagePredictor with the model path, resize size, 
        target classes, and preprocessing function.

        Args:
            model_path (str): Path to the trained model file.
            resize_size (tuple): Size to which the input images will be resized.
            targets (list): List of target class names.
            pre_processing_function: Function to preprocess the input images.
        """
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path)  # Load the model
        self.resize_size = resize_size
        self.targets = targets  # Add targets for prediction

    @classmethod
    def init_from_config_path(cls, config_path):
        """
        Initialize an ImagePredictor instance from a configuration YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            ImagePredictor: An instance of ImagePredictor.
        """
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        
        return cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"]  # Load target classes from config
        )

    def predict_from_array(self, arr):
        """
        Make a prediction using an image array.

        Args:
            arr (np.ndarray): The input image as a numpy array.

        Returns:
            dict: A dictionary containing the predicted class name.
        """
        input_shape = (150,150)  # Define the input shape for the model
        arr = resize_img(arr, input_shape)  # Resize the input image
        arr = self.pre_processing_function(arr)  # Preprocess the image
        
        # Make prediction
        pred = self.model.predict(arr[np.newaxis, ...]).ravel()
        class_index = np.argmax(pred)  # Get the index of the highest prediction
        predicted_class_name = self.targets[class_index]  # Map index to class name

        return {"predicted_class": predicted_class_name}

    def predict_from_file(self, file_path):
        """
        Make a prediction using an image file.

        Args:
            file_path (str): Path to the input image file.

        Returns:
            dict: A dictionary containing the predicted class name.
        """
        arr = read_from_file(file_path)  # Read the image from file
        return self.predict_from_array(arr)  # Make prediction from the array

def resize_img(image_array, shape):
    """
    Resize an image to the specified shape.

    Args:
        image_array (np.ndarray): The input image as a numpy array.
        shape (tuple): The target size for the image.

    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    img = Image.fromarray(image_array)  # Convert numpy array to PIL Image
    img = img.resize(shape)  # Resize the image
    return np.array(img)  # Convert back to numpy array

def read_from_file(file_path):
    """
    Read an image from a file and convert it to a numpy array.

    Args:
        file_path (str): Path to the input image file.

    Returns:
        np.ndarray: The image as a numpy array in RGB format.
    """
    img = Image.open(file_path)  # Open the image file
    img = img.convert("RGB")  # Ensure the image is in RGB format
    return np.array(img)  # Convert the image to a numpy array
