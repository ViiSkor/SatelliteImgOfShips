# [DS Basics | Project sample] Satellite Images Of Ships
This project is a example of basic data science project.

#### - Project Status: [In Progress]

## Project Objective
The Airbus Ship Detection Challenge aims to develop advanced machine learning solutions for accurately detecting and classifying ships in satellite imagery. This technology has the potential to bolster maritime security, protect marine ecosystems, and improve trade efficiency, contributing to a safer and more sustainable future for our oceans and society.
### Methods Used
Implemented approach is pretty simple.
* Semantic Segmentation
* Undersampling
* Augmentation
* BatchNormalization
* Dice + BCE Loss

### Technologies
* Python 3.9
* Pandas, Dast
* Numpy, Scipy
* Tensorflow 2.12.0

## Project Description
**Data Sources:** The project relies on a comprehensive dataset of satellite images, provided by Airbus, encompassing a wide range of maritime environments.

**Modeling Work:** Deep learning model, such as U-Net, is being utilized to tackle the ship detection challenge. These models are fine-tuned and optimized to achieve high IoU.

**Blockers and Challenges:** The project faces several challenges, including the need to handle a large volume of data effectively, ensuring model generalization across diverse maritime environments, resolving imbalanced data issue, and addressing computational resource limitations for training and inference.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](https://www.kaggle.com/competitions/airbus-ship-detection/data).
3. You could use kaggle notebook, in this case you don't need to download data, just connect it to this [notebook]((https://github.com/ViiSkor/SatelliteImgOfShips/blob/master/notebooks/kaggle-notebook.ipynb)).
4. But, firstly load this notebook at Kaggle.

#### Run loccaly
### Setup
```
cd SatelliteImgOfShips
python -m venv dst-env
source dst-env/bin/activate
pip install -r requirements.txt
```

### Run training
```
import os
import yaml

import albumentations as A
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from core.data.read_data import get_metadata
from core.data.preprocessing import undersample, SemanticSegmentationDataGenerator, get_train_val_sets
from core.model.UNet import init_model
from core.vis import show_loss, visualize_preds
from core.utils import seed_everything
from core.callbacks import get_callbacks
from core.inference import create_full_res_model
from core.metrics import dice_coef
from core.train import train

tf.config.run_functions_eagerly(True)

with open('SatelliteImgOfShips/config.yml', 'r') as yamlfile:
    config = yaml.load(yamlfile.read(), Loader=yaml.FullLoader)
    
seed_everything(config['meta']['seed'])

ship_dir = '../input/airbus-ship-detection'
mask_dir = os.path.join(ship_dir, 'train_ship_segmentations_v2.csv')
train_image_dir = os.path.join(ship_dir, 'train_v2')

balanced_df = undersample(unique_img_ids, samples_per_group=2000)
train_df, valid_df = get_train_val_sets(balanced_df, masks)

seg_model = init_model(config['model'])

transform = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=45, shift_limit=0.1, scale_limit=[0.9, 1.25]),
        ], p=0.4)

callbacks = get_callbacks('segmentation_model', config['train'])
seg_model, loss_history = train(seg_model, train_image_dir, config['train'], config['preprocessing']['img_scaling'], callbacks, train_df, valid_df, transform)
```

#### Testing
To run the test, install pytest using pip or conda and then from the repository root run
 
    pytest tests/test_model.py

#### Linting
To verify that your code adheres to python standards run linting as shown below:

    pylint core/*


## Featured Notebooks
* [EDA Notebook](https://github.com/ViiSkor/SatelliteImgOfShips/blob/master/notebooks/airbus-eda.ipynb)
* [Kaggle Notebook](https://github.com/ViiSkor/SatelliteImgOfShips/blob/master/notebooks/kaggle-notebook.ipynb)

## Project Contents

```
├── .gitignore               <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if 
│                               needed
├── config.yml               <- Config with hyperparameters and meto info.
├── README.md                <- The top-level README for developers using this project.
├── requirements.txt         <- The requirements file for reproducing the environment.
│
├── notebooks                <- Notebooks for analysis and testing
│   ├── airbus-eda           <- Notebooks for EDA
│   └── kaggle-notebook      <- Notebooks for Kaggle notebook that was used to run this project and inference testing.
│
├── core                     <- Code for use in this project.
│   ├── data                 <- Example python package - place shared code in such a package
│   ├── ├── __init__.py      <- Python package initialisation
│   │   ├── preprocessing.py <- Scripts for preprocessing and data generator
│   │   └── read_data.py     <- Script that read and clean raw data
│   ├── model                <- Example python package - place shared code in such a package
│   │   ├── __init__.py      <- Python package initialisation
│   │   ├── blocks.py        <- U-Net blocks implementation
│   │   └── UNet.py          <- U-Net initialization
│   ├── __init__.py          <- Python package initialisation
│   ├── callbacks.py         <- Example python package - place shared code in such a package
│   ├── inference.py         <- Method for inference
│   ├── losses.py            <- Implementation of Dice + BCE loss
│   ├── metrics.py           <- Implementation of dice metric
│   ├── utils.py             <- Utility methods
│   └── vis.py               <- Visualization methods for losses and predictions
│
└── tests                    <- Test cases (named after module)
    └── test_model.py        <- Example testing with check on overfitting ability.
```