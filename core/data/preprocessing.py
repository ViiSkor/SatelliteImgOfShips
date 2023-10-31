import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import albumentations as A
from skimage.io import imread
from sklearn.model_selection import train_test_split

from core.utils import masks_as_image


def undersample(unique_img_ids: pd.DataFrame, samples_per_group: int) -> pd.DataFrame:
    return unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(samples_per_group) if len(x) > samples_per_group else x)


def get_train_val_sets(balanced_df: pd.DataFrame, masks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, valid_ids = train_test_split(
        balanced_df, test_size=0.2, stratify=balanced_df['ships']
    )
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)
    return train_df, valid_df


def create_img_gen(
        in_df: pd.DataFrame,
        img_dir,
        img_scaling: Optional[tuple[int, int]],
        shuffle_batches: bool=True,
        do_augmentation: bool=True,
        transform: Optional[A.Compose]=None
) -> Tuple[list, list]:
    all_batches = list(in_df.groupby('ImageId'))

    while True:
        if shuffle_batches:
            np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(img_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)

            if img_scaling is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]

            if do_augmentation:
                transformed = transform(image=c_img, mask=c_mask)
                c_img = transformed['image']
                c_mask = transformed['mask']

            yield c_img / 255.0, c_mask
