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
* Dice + BSE Loss

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
2. Raw Data is being kept [here].

3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)


#### Setup using
```
cd SatelliteImgOfShips
python -m venv dst-env
```

#### Activate environment
Max / Linux
```
source dst-env/bin/activate
```

#### Install Dependencies
```
pip install -r requirements.txt
```

    
#### Testing
To run the test, install pytest using pip or conda and then from the repository root run
 
    pytest tests/test_model.py

#### Linting
To verify that your code adheres to python standards run linting as shown below:

    pylint core/*


## Featured Notebooks
* [EDA Notebook](link)
* [Kaggle Notebook](link)

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
│   └── Kaggle_Notebook      <- Notebooks for Kaggle notebook that was used to run this project and inference testing.
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