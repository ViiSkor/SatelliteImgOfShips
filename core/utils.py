import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf
from skimage.util import montage
from skimage.morphology import label

DEFAULT_RANDOM_SEED = 42


def seed_basic(seed: int=DEFAULT_RANDOM_SEED):
    """
    Set random seeds for basic Python, NumPy, and hash seed.

    This function sets random seeds for Python's built-in random module, NumPy, and Python's hash seed for reproducible
    randomness.

    Parameters:
    - seed (int): The random seed to be set. Default is 42.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_tf(seed: int=DEFAULT_RANDOM_SEED):
    """
    Set the random seed for TensorFlow.

    This function sets the random seed for TensorFlow to ensure reproducible results.

    Parameters:
    - seed (int): The random seed to be set. Default is 42.
    """

    tf.random.set_seed(seed)


def seed_everything(seed: int=DEFAULT_RANDOM_SEED):
    """
    Set random seeds for various libraries.

    This function sets random seeds for Python's built-in random module, NumPy, Python's hash seed, and TensorFlow
    for consistent and reproducible results.

    Parameters:
    - seed (int): The random seed to be set. Default is 42.
    """

    seed_basic(seed)
    seed_tf(seed)


def multi_rle_encode(img: np.array, **kwargs) -> list[str]:
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [
            rle_encode(np.sum(labels == k, axis=2), **kwargs) for k in np.unique(labels[labels > 0])
        ]
    return [rle_encode(labels == k, **kwargs) for k in np.unique(labels[labels > 0])]


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(
        img: np.array,
        min_max_threshold: float=1e-3,
        max_mean_threshold: Optional[float]=None
) -> str:
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> np.array:
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list: list, img_shape: tuple[int, int] = (768, 768)) -> np.array:
    """
    Create a binary mask array from a list of run-length-encoded (RLE) masks.

    This function takes a list of run-length-encoded masks and combines them into a single binary mask array where
    each pixel is set to 1 if any of the masks in the list indicate a pixel in the object.

    Parameters:
    - in_mask_list (list): A list of run-length-encoded masks (strings).
    - img_shape (tuple): The shape of the output binary mask array. Default is (768, 768).

    Returns:
    - np.array: A binary mask array where 1 indicates the presence of an object.
    """

    all_masks = np.zeros(img_shape, dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def scale(x, in_mask_list):
    """
    Scale a heatmap image to shift its intensity.

    This function scales a heatmap image to shift its intensity for visualization purposes. It is used to create
    a color mask array for each ship in the input list of run-length-encoded masks.

    Parameters:
    - x: A value representing the index of the current mask being processed.
    - in_mask_list (list): A list of run-length-encoded masks.

    Returns:
    - float: The scaled value used to adjust the intensity of the heatmap.
    """

    return (len(in_mask_list) + x + 1) / (len(in_mask_list) * 2)


def masks_as_color(in_mask_list: list, img_shape: tuple[int, int] = (768, 768)) -> np.array:
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros(img_shape, dtype=np.float)
    for i, mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:, :] += scale(i, in_mask_list) * rle_decode(mask)
    return all_masks


def montage_rgb(x: np.array) -> np.array:
    return np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
