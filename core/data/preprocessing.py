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


class SemanticSegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, in_df, img_dir, batch_size, img_scaling=None, shuffle_batches=True, do_augmentation=False,
                 transform=None):
        self.in_df = in_df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_scaling = img_scaling
        self.shuffle_batches = shuffle_batches
        self.do_augmentation = do_augmentation
        self.transform = transform

        self.all_batches = list(self.in_df.groupby('ImageId'))
        self.out_rgb, self.out_mask = [], []

    def __len__(self):
        return len(self.in_df) // self.batch_size

    def __getitem__(self, index):
        for c_img_id, c_masks in self.all_batches:
            rgb_path = os.path.join(self.img_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)

            if self.img_scaling is not None:
                c_img = c_img[::self.img_scaling[0], ::self.img_scaling[1]]
                c_mask = c_mask[::self.img_scaling[0], ::self.img_scaling[1]]

            if self.do_augmentation and self.transform is not None:
                transformed = self.transform(image=c_img, mask=c_mask)
                c_img = transformed['image']
                c_mask = transformed['mask']

            self.out_rgb.append(c_img)
            self.out_mask.append(c_mask)

            if len(self.out_rgb) >= self.batch_size:
                batch_rgb = tf.convert_to_tensor(np.stack(self.out_rgb, 0) / 255.0, dtype=tf.float32)
                batch_mask = tf.convert_to_tensor(np.stack(self.out_mask, 0), dtype=tf.float32)
                self.out_rgb.clear(), self.out_mask.clear()
                return batch_rgb, batch_mask

        def on_epoch_end(self):
            if self.shuffle_batches:
                np.random.shuffle(self.all_batches)
