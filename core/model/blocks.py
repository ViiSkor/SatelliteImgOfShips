from typing import Callable, Union, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def upsample_conv(
        filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: Union['valid', 'same']
) -> tf.Tensor:
    """
    Create an Conv2DTranspose layer.

    Parameters:
    - filters (int): Number of output filters.
    - kernel_size (Tuple[int, int]): Size of the convolutional kernel.
    - strides (Tuple[int, int]): Stride values for the convolution operation.
    - padding (str): Padding mode, either 'valid' or 'same'.

    Returns:
    - tf.Tensor: Output tensor after applying the upsample convolutional layer.
    """

    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(
        filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: Union['valid', 'same']
) -> tf.Tensor:
    """
    Create a UpSampling2D layer.

    Parameters:
    - filters (int): Number of output filters.
    - kernel_size (Tuple[int, int]): Size of the upsampling kernel.
    - strides (Tuple[int, int]): Stride values for the upsampling operation.
    - padding (str): Padding mode, either 'valid' or 'same'.

    Returns:
    - tf.Tensor: Output tensor after applying the simple upsampling layer.
    """

    return layers.UpSampling2D(strides)


def encoder_block(prev_layer_inputs: tf.Tensor, n_filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create an encoder block consisting of convolutional layers and pooling.

    Parameters:
    - prev_layer_inputs (tf.Tensor): Input tensor to the encoder block.
    - n_filters (int): Number of filters for the convolutional layers.

    Returns:
    - Tuple[tf.Tensor, tf.Tensor]: A tuple containing the pooled output tensor and the skip connection tensor.
    """

    conv = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(prev_layer_inputs)
    conv = layers.BatchNormalization()(conv)
    skip_connection = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv)
    pool = layers.MaxPooling2D((2, 2))(skip_connection)
    pool = layers.BatchNormalization()(pool)
    return pool, skip_connection


def decoder_block(
        prev_layer_input: tf.Tensor, skip_layer_input: tf.Tensor, n_filters: int, upsample: Callable
) -> tf.Tensor:
    """
    Create a decoder block consisting of upsampling and convolutional layers.

    Parameters:
    - prev_layer_input (tf.Tensor): Input tensor to the decoder block.
    - skip_layer_input (tf.Tensor): Skip connection tensor from the encoder block.
    - n_filters (int): Number of filters for the convolutional layers.
    - upsample (Callable): The upsampling function to use.

    Returns:
    - tf.Tensor: Output tensor after applying the decoder block.
    """

    up = upsample(n_filters, (2, 2), strides=(2, 2), padding='same')(prev_layer_input)
    up = layers.concatenate([up, skip_layer_input])
    up = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up)
    up = layers.BatchNormalization()(up)
    up = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up)
    up = layers.BatchNormalization()(up)
    return up
