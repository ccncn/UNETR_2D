#######################################
## Author: Aitor González (@AAitorG) ##
#######################################

from math import log2
from tensorflow.keras import Model, layers
from .modules import *

def UNETR_2D(
            input_shape,
            patch_size,
            num_patches,
            projection_dim,
            transformer_layers,
            num_heads,
            transformer_units,
            data_augmentation = None,
            num_filters = 16, 
            num_classes = 1,
            decoder_activation = 'relu',
            decoder_kernel_init = 'he_normal',
            ViT_hidd_mult = 3,
            batch_norm = True,
            dropout = 0.0,
            pretrained_vit_encoder = None  # NEW: pass SimCLR-pretrained encoder weights
        ):

    """
    UNETR architecture adapted for 2D operations. It combines a ViT with U-Net, replaces the convolutional encoder with the ViT
    and adapt each skip connection signal to their layer's spatial dimensionality. 

    Args (updated):
      pretrained_vit_encoder: optional pretrained ViT encoder (e.g., SimCLR-trained).
    """

    # ViT

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs) if data_augmentation != None else inputs
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    hidden_states_out = []

    # Apply SimCLR pretrained encoder if provided
    if pretrained_vit_encoder:
        x = pretrained_vit_encoder(encoded_patches)
        encoded_patches = x[-1]
        hidden_states_out = x[:-1]  # assumes intermediate layers are returned
    else:
        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])
            hidden_states_out.append(encoded_patches)

    # UNETR Part (bottom_up)

    total_upscale_factor = int(log2(patch_size))
    if type(dropout) is float: 
        dropout = [dropout,]*total_upscale_factor

    z = layers.Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])(encoded_patches)
    x = up_green_block(z, num_filters * (2**(total_upscale_factor-1)))

    for layer in reversed(range(1, total_upscale_factor)):
        z = layers.Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])(
            hidden_states_out[ (ViT_hidd_mult * layer) - 1 ]
        )
        for _ in range(total_upscale_factor - layer):
            z = mid_blue_block(z, num_filters * (2**layer), activation=decoder_activation,
                               kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        x = layers.concatenate([x, z])
        x = two_yellow(x, num_filters * (2**(layer)), activation=decoder_activation,
                       kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        x = up_green_block(x, num_filters * (2**(layer-1)))

    first_skip = two_yellow(augmented, num_filters, activation=decoder_activation,
                            kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0]) 
    x = layers.concatenate([first_skip, x])

    x = two_yellow(x, num_filters, activation=decoder_activation,
                   kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0])
    output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name="mask")(x)

    model = Model(inputs=inputs, outputs=output)
    return model
