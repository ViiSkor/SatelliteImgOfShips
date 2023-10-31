# Model Profile

## Overview
- **Name:** U-Net
- **Type:** Convolutional Neural Network (CNN)
- **Date of Introduction:** 2015

## Key Characteristics

### Augmentations
```
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.0, p=0.5),
        A.IAAPerspective(p=0.2),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=0.1),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.Sharpen (p=0.4),
                A.Blur(blur_limit=3, p=0.4),
                A.MotionBlur(blur_limit=3, p=0.4),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.7),
                A.HueSaturationValue(p=0.7),
            ],
            p=0.4,
        ),
    ])
```

### Loss Function
- 1e-3 * Binary Cross-entropy loss - Dice Loss

### Data Preprocessing
- Undersampling
- Normalization
- Del corrupted files
- Del files less than 50kb

### Training
- Training config is [here](https://github.com/ViiSkor/SatelliteImgOfShips/blob/master/config.yml)

### Final Scores
| Loss    | Dice   | Binary Acc | POD    | Epochs |
|---------|--------|------------|--------|--------|
| -0.3859 | 0.3688 | 0.9287     | 0.6467 | 70     |

### Relevant commit for the experiment
```4da907add32589e241642892ab0560387375422a```
