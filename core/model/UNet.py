from tensorflow.keras import models, layers
from core.model.blocks import upsample_conv, upsample_simple, encoder_block, decoder_block


def init_model(config: dict, input_shape: tuple=(256, 256, 3)) -> models.Model:
    """
    Initialize and build a U-Net-style image segmentation model.

    This function constructs a U-Net architecture for image segmentation based on the provided configuration.

    Parameters:
    - config (dict): A dictionary containing model configuration settings.
    - input_shape (tuple): Tuple specifying the input image shape, default is (256, 256, 3).

    Returns:
    - models.Model: A Keras Model representing the U-Net-style image segmentation model.

    Configuration Settings:
    - 'n_filters' (int): Number of initial filters in the model.
    - 'upsample_mode' (str): Upsampling mode, either 'DECONV' for deconvolution or 'SIMPLE' for simple upsampling.
    - 'net_scaling' (int, None): Scale factor for downscaling and subsequent upscaling of the network.
    If None, no scaling is applied.
    - 'gaussian_noise' (float): Standard deviation of Gaussian noise applied to the input.

    Example Configuration:
    config = {
        'n_filters': 64,
        'upsample_mode': 'DECONV',
        'net_scaling': 2,
        'gaussian_noise': 0.01
    }
    """

    n_filters = config['n_filters']

    if config['upsample_mode'] == 'DECONV':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    input_img = layers.Input(input_shape, name='RGB_Input')
    pp_in_layer = input_img

    if config['net_scaling'] is not None:
        pp_in_layer = layers.AvgPool2D(config['net_scaling'])(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(config['gaussian_noise'])(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    enc1 = encoder_block(pp_in_layer, n_filters)
    enc2 = encoder_block(enc1[0], n_filters * 2)
    enc3 = encoder_block(enc2[0], n_filters * 4)
    enc4 = encoder_block(enc3[0], n_filters * 8)
    enc5 = encoder_block(enc4[0], n_filters * 16)

    c6 = layers.Conv2D(n_filters * 32, (3, 3), activation='relu', padding='same')(enc5[0])
    c6 = layers.Conv2D(n_filters * 32, (3, 3), activation='relu', padding='same')(c6)

    dec1 = decoder_block(c6, enc5[1], n_filters * 16, upsample)
    dec2 = decoder_block(dec1, enc4[1], n_filters * 8, upsample)
    dec3 = decoder_block(dec2, enc3[1], n_filters * 4, upsample)
    dec4 = decoder_block(dec3, enc2[1], n_filters * 2, upsample)
    dec5 = decoder_block(dec4, enc1[1], n_filters, upsample)

    dec = layers.Conv2D(1, (1, 1), activation='sigmoid')(dec5)

    if config['net_scaling'] is not None:
        dec = layers.UpSampling2D(config['net_scaling'])(dec)

    model = models.Model(inputs=[input_img], outputs=[dec])
    return model
