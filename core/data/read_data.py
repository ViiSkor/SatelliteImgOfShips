import os
from typing import Tuple

import numpy as np
import pandas as pd
import dask.dataframe as dd


def get_metadata(mask_dir: str, npartitions: int=8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    masks = pd.read_csv(mask_dir)
    not_empty = pd.notna(masks.EncodedPixels)
    masks = dd.from_pandas(masks, npartitions=npartitions)
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

    # some files are too small/corrupt
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size / 1024, meta=('ImageId', int))

    unique_img_ids = unique_img_ids.compute()
    masks = masks.compute()

    masks.drop(['ships'], axis=1, inplace=True)

    return masks, unique_img_ids
