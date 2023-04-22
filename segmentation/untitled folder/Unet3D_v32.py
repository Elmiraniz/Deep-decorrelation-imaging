# Author: Elmira Ghahramani Z.
# Date: 03/07/2023
# Last update: 04/10/2023
########################################################################################
print('Running v32 ...')
# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import concatenate

from generator import customImageDataGenerator

import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

########################################################################################
# Unet model

# Encoder
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv3D(filters=n_filters,  # Number of filters
                  kernel_size=3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv3D(filters=n_filters,  # Number of filters
                  kernel_size=3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(rate=dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling3D(pool_size=2, strides=2)(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

# Decoder
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """

    up = Conv3DTranspose(
        filters=n_filters,    # number of filters
        kernel_size=3,    # Kernel size
        strides=2,
        padding='same')(expansive_input)

    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=4)
    conv = Conv3D(filters=n_filters,   # Number of filters
                  kernel_size=3,     # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv3D(filters=n_filters,   # Number of filters
                  kernel_size=3,     # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv

# Build the model
def unet_3D_model(input_size=(64, 64, 128, 1), n_filters=32, n_classes=1):
    """
    Unet model

    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)

    if input_size[2] == 128:
        cblock0 = Conv3D(filters=n_filters,  # Number of filters
                         kernel_size=[1, 1, 65],   # Kernel size
                         padding='valid',
                         kernel_initializer='he_normal')(inputs)
        # Contracting Path (encoding)
        # Add a conv_block with the inputs of the unet_ model and n_filters
        cblock1 = conv_block(cblock0, n_filters)
    elif input_size[2] == 64:
        # Contracting Path (encoding)
        # Add a conv_block with the inputs of the unet_ model and n_filters
        cblock1 = conv_block(inputs, n_filters)

    # Chain the first element of the output of each block to be the input of the next conv_block.
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 4*n_filters)
    # Include a dropout_prob of 0.3 for this layer
    cblock4 = conv_block(cblock3[0], 8*n_filters, dropout_prob=0.3)
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], 16*n_filters,
                         dropout_prob=0.3, max_pooling=False)

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8*n_filters)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1], 4*n_filters)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2*n_filters)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv3D(n_filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv10 = Conv3D(n_classes, kernel_size=1,
                    activation='sigmoid', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_DEPTH = 64
N_CHANNELS = 1

unet3D = unet_3D_model((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, N_CHANNELS))
unet3D.summary()

# Loss function
unet3D.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.TrueNegatives(),
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.FalseNegatives()])
########################################################################################
# Load data and convert to tf dataset
temp1 = np.load('../Bmode_global_combined_cumdec_and_mask.npz')
temp2 = [temp1[key] for key in temp1]

Bmode_64 = np.array(temp2[0])
global_cumdec_64 = np.array(temp2[1])
combined_cumdec_64 = np.array(temp2[2])
mask_64 = np.array(temp2[3])

# normalize data
def decorr_scale(x):
    MAX_DECORR = 0.08
    DYN_RANGE = 3
    
    x[x<=0] = np.exp(-16)
    x = (np.log10(x/MAX_DECORR) + DYN_RANGE) / DYN_RANGE
    x[x<0] = 0
    x[x>1] = 1
    return x

combined_cumdec_64_log = decorr_scale(combined_cumdec_64)

print(f'Max comb cumdec log={np.max(np.ravel(combined_cumdec_64_log))}')

# Create datasets
# image_64 = np.concatenate((Bmode_64, global_cumdec_64), axis=3)
# image_64 = np.concatenate((Bmode_64, combined_cumdec_64_log), axis=3)
image_64 = combined_cumdec_64_log

trials_to_train = 42
trials_to_validate = 56
########################################################################################
# Train the network
EPOCHS = 200
BATCH_SIZE = 10

def generator(images, groundtruth, batch):
    """Load a batch of augmented images"""
  
    gen = customImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1,
                                   shear_range=0.174,
                                   zoom_range=(0.8,1.2),
                                   fill_mode='constant',
                                   cval=0,
                                   horizontal_flip=True
                                   )

    for b in gen.flow(x=images, y=groundtruth, batch_size=batch):
        yield (b[0], np.round((b[1]).astype(float)))

def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.01)    

learningrate_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1) 

model_history = unet3D.fit(
    x=generator(combined_cumdec_64_log[:trials_to_train,...,tf.newaxis], mask_64[:trials_to_train,...,tf.newaxis], BATCH_SIZE),
    validation_data=(combined_cumdec_64_log[trials_to_train:trials_to_validate,...,tf.newaxis], mask_64[trials_to_train:trials_to_validate,...,tf.newaxis]),
    steps_per_epoch=trials_to_train / BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[learningrate_scheduler_callback])

OUTPUT_PATH = 'saved models/'
unet3D.save(OUTPUT_PATH + 'Unet3D_v32.h5')
np.save(OUTPUT_PATH + 'Unet3D_v32_history.npy', model_history.history)
