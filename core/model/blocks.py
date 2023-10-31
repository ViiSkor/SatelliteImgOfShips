from typing import Callable, Union, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def upsample_conv(
        filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: Union['valid', 'same']
) -> tf.Tensor:
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(
        filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: Union['valid', 'same']
) -> tf.Tensor:
    return layers.UpSampling2D(strides)


def encoder_block(prev_layer_inputs: tf.Tensor, n_filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    conv = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(prev_layer_inputs)
    conv = layers.BatchNormalization()(conv)
    skip_connection = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv)
    pool = layers.MaxPooling2D((2, 2))(skip_connection)
    pool = layers.BatchNormalization()(pool)
    return pool, skip_connection


def decoder_block(
        prev_layer_input: tf.Tensor, skip_layer_input: tf.Tensor, n_filters: int, upsample: Callable
) -> tf.Tensor:
    up = upsample(n_filters, (2, 2), strides=(2, 2), padding='same')(prev_layer_input)
    up = layers.concatenate([up, skip_layer_input])
    up = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up)
    up = layers.BatchNormalization()(up)
    up = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up)
    up = layers.BatchNormalization()(up)
    return up
