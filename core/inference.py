import os
from typing import Tuple

import numpy as np
from skimage.io import imread
from skimage.morphology import disk, binary_opening
from tensorflow.keras import models, layers


def create_full_res_model(model, img_scaling: tuple[int, int]):
    """
    Create a full-resolution model by adding scaling operations to the output of an existing model.

    This function takes an existing Keras model and, if img_scaling is not None, creates a new model that scales the
    input image, applies the provided model, and then scales the output back to full resolution.

    Parameters:
    - model: The Keras model to which scaling operations will be applied.
    - img_scaling (Tuple[int, int]): Tuple specifying the scaling factors for height and width. Use None for no scaling.

    Returns:
    - models.Model: A Keras model with scaling operations applied if img_scaling is not None, or the original model.
    """

    if img_scaling is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(img_scaling, input_shape=(None, None, 3)))
        fullres_model.add(model)
        fullres_model.add(layers.UpSampling2D(img_scaling))
        return fullres_model

    return model


def raw_prediction(model, c_img_name : str, path: str) -> np.array:
    """
     Perform raw image segmentation using the provided model.

     This function reads an image from a specified path, preprocesses it, and uses the model to perform
     image segmentation.

     Parameters:
     - model: The Keras model used for image segmentation.
     - c_img_name (str): Name of the image file to be processed.
     - path (str): The directory path where the image is located.

     Returns:
     - np.array: The raw segmentation result.
     - np.array: The input image.
     """

    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg: np.array) -> np.array:
    """
    Apply morphological operations to smooth the segmentation mask.

    This function applies binary opening to the provided segmentation mask.

    Parameters:
    - cur_seg (np.array): The input binary segmentation mask.

    Returns:
    - np.array: The smoothed binary segmentation mask.

    """

    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def predict(model, img: np.array, path) -> Tuple[np.array, np.array]:
    cur_seg, c_img = raw_prediction(model, img, path=path)
    return smooth(cur_seg), c_img
