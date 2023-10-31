from typing import Tuple
from pathlib import Path

import yaml
import pytest
import numpy as np
from skimage.draw import rectangle
from tensorflow.keras.optimizers import AdamW

from core.losses import dice_p_bce
from core.metrics import dice_coef
from core.model.UNet import init_model


@pytest.fixture()
def toy_data() -> Tuple[np.array, np.array]:
    # Create a blank image
    image = np.zeros((256, 256, 3))

    # Create a white rectangle in the image
    rr, cc = rectangle(start=(50, 50), extent=(100, 100))
    image[rr, cc, :] = 1

    # Create a binary mask (segmentation ground truth)
    mask = np.zeros_like(image)
    mask[rr, cc] = 1
    return image, mask


@pytest.fixture()
def config() -> dict:
    with open('../SatelliteImgOfShips/config.yml', 'r') as yamlfile:
        config = yaml.load(yamlfile.read(), Loader=yaml.FullLoader)
    return config


@pytest.fixture()
def loss_from_one_batch_trained_model(config: dict, toy_data: Tuple[np.array, np.array]):
    model = init_model(config['model'])
    model.compile(optimizer=AdamW(learning_rate=0.001), loss=dice_p_bce, run_eagerly=True)

    loss_history = model.fit(
        np.array([toy_data[0], toy_data[0]]),
        np.array([toy_data[1], toy_data[1]]),
        epochs=10,
        verbose=1
    )
    return loss_history



def test_overfit_batch(loss_from_one_batch_trained_model):
    train_loss = loss_from_one_batch_trained_model.history['loss']
    print(train_loss)

    assert train_loss[-1] < -0.5
