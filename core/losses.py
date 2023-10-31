import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from core.metrics import dice_coef


def dice_p_bce(y_true, y_pred):
    """
    Compute the Dice coefficient-based loss with an added binary cross-entropy term.

    This loss function combines the Dice coefficient loss and binary cross-entropy loss, providing a weighted sum
    of both terms to optimize a model for binary image segmentation tasks.

    Parameters:
    - y_true (tf.Tensor): The true binary ground truth mask.
    - y_pred (tf.Tensor): The predicted binary segmentation mask.

    Returns:
    - tf.Tensor: The combined loss value, which is a weighted sum of binary cross-entropy and negative Dice coefficient.
    """

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return 1e-3 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
