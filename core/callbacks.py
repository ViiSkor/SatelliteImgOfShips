from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def get_callbacks(model_name: str, config: dict) -> list:
    """
    Get a list of Keras callback objects for model training.

    This function creates a list of callback objects to be used during the training of a Keras model.

    Parameters:
    - model_name (str): Name of the model, used to generate checkpoint file names.
    - config (dict): A dictionary containing configuration settings for the callbacks.

    Returns:
    - list: A list of Keras callback objects.

    Configuration Settings:
    - 'checkpoint' (dict): Configuration settings for ModelCheckpoint.
    - 'reduceLROnPlat' (dict): Configuration settings for ReduceLROnPlateau.
    - 'EarlyStopping' (dict): Configuration settings for EarlyStopping.

    Example Configuration:
    config = {
        'checkpoint': {
            'save_best_only': True,
            'save_weights_only': True,
            'mode': 'min'
        },
        'reduceLROnPlat': {
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
        'EarlyStopping': {
            'patience': 10,
            'min_delta': 0.0001
        }
    }
    """

    weight_path=f'{model_name}_weights.best.hdf5'
    checkpoint = ModelCheckpoint(
        weight_path, monitor='val_loss', verbose=1, mode='min', **config['checkpoint']
    )
    reduce_lr_on_plat = ReduceLROnPlateau(
        monitor='val_loss', verbose=1, mode='min', **config['reduceLROnPlat']
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', verbose=2, **config['EarlyStopping']
    )
    return [checkpoint, early_stopping, reduce_lr_on_plat]
