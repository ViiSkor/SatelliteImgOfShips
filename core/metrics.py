import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient for binary image segmentation.

    The Dice coefficient is a measure of the similarity between the true binary mask and the predicted binary mask.
    It quantifies the overlap between the two masks.

    Parameters:
    - y_true (tf.Tensor): The true binary ground truth mask.
    - y_pred (tf.Tensor): The predicted binary segmentation mask.
    - smooth (float): A smoothing factor to prevent division by zero. Default is 1.

    Returns:
    - tf.Tensor: The computed Dice coefficient.
    """

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def POD(y_true, y_pred):
    """
    Compute the Probability of Detection (POD) for binary classification.

    The POD measures the ability of a model to correctly detect positive cases.
    It is commonly used in binary classification tasks.

    Parameters:
    - y_true (tf.Tensor): The true binary ground truth mask.
    - y_pred (tf.Tensor): The predicted binary segmentation mask.

    Returns:
    - tf.Tensor: The computed Probability of Detection (POD).
    """

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    return true_pos / (true_pos + false_neg)
