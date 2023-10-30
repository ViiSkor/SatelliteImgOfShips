import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import albumentations as A
from skimage.io import imread


def undersample(unique_img_ids: pd.DataFrame, samples_per_group: int) -> pd.DataFrame:
    return unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(samples_per_group) if len(x) > samples_per_group else x)


def create_img_gen(
        in_df: pd.DataFrame,
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
            rgb_path = os.path.join(train_image_dir, c_img_id)
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
