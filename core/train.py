from tensorflow.keras.optimizers import AdamW
from core.data.preprocessing import SemanticSegmentationDataGenerator
from core.losses import dice_p_bce
from core.metrics import POD, dice_coef


def train(model, img_dir, config, img_scaling, callbacks, train_df, valid_df, transform):
    """
    Train a semantic segmentation model using the specified configuration and data.

    This function trains a semantic segmentation model using the provided Keras model, data directories,
    configuration settings, and data generators. It compiles the model with a custom loss function, metrics,
    and the AdamW optimizer.

    Parameters:
    - model: The Keras model to be trained for semantic segmentation.
    - img_dir (str): The directory path where image data is located.
    - config (dict): A dictionary containing configuration settings for training.
    - img_scaling (tuple): A tuple specifying the scaling factors for height and width of input images.
    - callbacks (list): A list of Keras callback objects for monitoring and controlling the training process.
    - train_df (DataFrame): A Pandas DataFrame containing training dataset information.
    - valid_df (DataFrame): A Pandas DataFrame containing validation dataset information.
    - transform: A function or callable for data augmentation.

    Returns:
    - model: The trained Keras model.
    - loss_history: The training history, including loss and metric values.

    Configuration Settings:
    - 'batch_size' (int): Batch size for training and validation.
    - 'learning_rate' (float): Learning rate for the AdamW optimizer.
    - 'do_augmentation' (bool): Whether to apply data augmentation during training.
    - 'epochs' (int): Number of training epochs.
    """

    batch_size = config['batch_size']
    model.compile(optimizer=AdamW(learning_rate=config['learning_rate']), loss=dice_p_bce,
                      metrics=['binary_accuracy', dice_coef, POD], run_eagerly=True)

    train_data_generator = SemanticSegmentationDataGenerator(
        train_df, img_dir, batch_size, img_scaling, config['do_augmentation'], transform
    )
    val_data_generator = SemanticSegmentationDataGenerator(valid_df, img_dir, batch_size, img_scaling)
    loss_history = model.fit(
        train_data_generator,
        steps_per_epoch=len(train_data_generator) // train_data_generator.batch_size,
        epochs=config['epochs'],
        validation_data=val_data_generator,
        validation_steps=len(val_data_generator) // val_data_generator.batch_size,
        callbacks=callbacks,
        verbose=1,
        workers=1,
    )

    return model, loss_history
