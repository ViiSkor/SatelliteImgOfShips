from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def get_callbacks(model_name: str, config: dict) -> list:
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
