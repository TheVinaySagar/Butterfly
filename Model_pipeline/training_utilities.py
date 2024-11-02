from sklearn.model_selection import train_test_split
import sys
import os.path
import numpy as np
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
# from Pipeline.preprocessing_utilities import read_img_from_path, resize_img
# class Rescaling(tensorflow.keras.layers.Layer): # type: ignore
#     """Multiply inputs by `scale` and adds `offset`.
#     For instance:
#     1. To rescale an input in the `[0, 255]` range
#     to be in the `[0, 1]` range, you would pass `scale=1./255`.
#     2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` 
#     range,
#     you would pass `scale=1./127.5, offset=-1`.
#     The rescaling is applied both during training and inference.
#     Input shape:
#     Arbitrary.
#     Output shape:
#     Same as input.
#     Arguments:
#     scale: Float, the scale to apply to the inputs.
#     offset: Float, the offset to apply to the inputs.
#     name: A string, the name of the layer.
#     """

#     def __init__(self, scale, offset=0., name=None, **kwargs):
#         self.scale = scale
#         self.offset = offset
#         super(Rescaling, self).__init__(name=name, **kwargs)

#     def call(self, inputs):
#         dtype = self._compute_dtype
#         scale = tensorflow.cast(self.scale, dtype)# type: ignore
#         offset = tensorflow.cast(self.offset, dtype)# type: ignore
#         return tensorflow.cast(inputs, dtype) * scale + offset# type: ignore

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = {
#             'scale': self.scale,
#             'offset': self.offset,
#         }
#         base_config = super(Rescaling, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataframe into training and validation sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)

def create_data_generators(train_df, val_df, train_dir, target_size, batch_size):
    """Creates image data generators for training and validation."""
    
    # Augmentation for training data
    train_aug_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Use VGG16 preprocessing
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling for validation data
    val_test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)  # Use VGG16 preprocessing

    # Training generator
    train_gen_vgg = train_aug_generator.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='filename',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Validation generator 
    val_gen_vgg = val_test_generator.flow_from_dataframe(
        dataframe=val_df,
        directory=train_dir,
        x_col='filename',
        y_col='label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_gen_vgg, val_gen_vgg

# # Usage Example
# train_df, val_df = split_data(df)
# train_dir = "Butterfly_Dataset/train"
# target_size = (128, 128)  # Example target size
# batch_size = 32  # Example batch size

# train_gen_vgg, val_gen_vgg = create_data_generators(train_df, val_df, train_dir, target_size, batch_size)
