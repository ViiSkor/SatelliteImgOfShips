from tensorflow.keras.optimizers import AdamW
from core.data.preprocessing import SemanticSegmentationDataGenerator
from core.losses import dice_p_bce
from core.metrics import POD, dice_coef


def train(model, img_dir, config, img_scaling, callbacks, train_df, valid_df, transform):
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
