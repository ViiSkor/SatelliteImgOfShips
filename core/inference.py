from typing import Tuple

import numpy as np
from skimage.morphology import disk, binary_opening
from tensorflow.keras import models, layers


def create_full_res_model(model, img_scaling: tuple[int, int]):
    if img_scaling is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(img_scaling, input_shape=(None, None, 3)))
        fullres_model.add(model)
        fullres_model.add(layers.UpSampling2D(img_scaling))
    else:
        fullres_model = seg_model
    return fullres_model


def raw_prediction(model, img: np.array, fullres_model, path: str) -> np.array:
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0) / 255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg, c_img[0]


def smooth(cur_seg: np.array) -> np.array:
    return binary_opening(cur_seg > 0.99, np.expand_dims(disk(2), -1))


def predict(model, img: np.array, path) -> Tuple[np.array, np.array]:
    cur_seg, c_img = raw_prediction(model, img, path=path)
    return smooth(cur_seg), c_img

