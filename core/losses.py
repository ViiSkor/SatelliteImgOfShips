import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from core.metrics import dice_coef


def dice_p_bce(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return 1e-3 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
