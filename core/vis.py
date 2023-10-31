import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from core.inference import smooth, raw_prediction
from core.utils import masks_as_color, multi_rle_encode


def show_loss(loss_history):
    """
    Display a plot of training and validation loss over epochs.

    This function takes the loss history from a model's training and displays a plot showing the training and validation
    loss over epochs, as well as binary accuracy.

    Parameters:
    - loss_history: A list of training history objects.
    """

    epochs = np.concatenate([mh.epoch for mh in loss_history])
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(
        epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
        epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-'
    )
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')


def visualize_preds(model, img_dir, masks, df):
    """
    Visualize model predictions, ground truth, and input images for a sample of images.

    This function takes a trained model, image directory, mask information, and a DataFrame containing image metadata.
    It visualizes the model's predictions, ground truth, and input images for a sample of images with
    different ship counts.

    Parameters:
    - model: The trained semantic segmentation model.
    - img_dir (str): The directory path where image data is located.
    - masks: A list of encoded masks for the images.
    - df: A DataFrame containing image metadata.
    """

    ## Get a sample of each group of ship count
    samples = df.groupby('ships').apply(lambda x: x.sample(1))
    _, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))
    _ = [c_ax.axis('off') for c_ax in m_axs.flatten()]

    for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
        first_seg, first_img = raw_prediction(model, c_img_name, img_dir)
        ax1.imshow(first_img)
        ax1.set_title('Image: ' + c_img_name)
        ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
        ax2.set_title('Model Prediction')
        reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
        ax3.imshow(reencoded)
        ax3.set_title('Prediction Masks')
        ground_truth = masks_as_color(
            masks.query(f'ImageId=="{c_img_name}"')['EncodedPixels']
        )
        ax4.imshow(ground_truth)
        ax4.set_title('Ground Truth')
