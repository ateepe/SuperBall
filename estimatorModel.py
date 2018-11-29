import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

import BoardGen as bg


# Include the empty tiles as an input?
# tiles = ['.', 'p', 'b', 'y', 'r', 'g']
tiles = ['p', 'b', 'y', 'r', 'g']

def build_model(channel_length=80, data_shape=(8, 10, 1)):
    """

    The shape of the input data has to be a tuple:
    (<batch_size>, <rows>, <cols>, <channels>)

    For this network, the channels are split across different input images.
    So each individual input image should have the shape (<rows>, <cols>, 1).

    """

    # Channel 1 - first binary tile color
    input_1 = Input(shape=data_shape)
    conv_1 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_1)
    drop_1 = Dropout(0.5)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_1)
    flat_1 = Flatten()(pool_1)

    # Channel 2 - second binary tile color
    input_2 = Input(shape=data_shape)
    conv_2 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_2)
    drop_2 = Dropout(0.5)(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_2)
    flat_2 = Flatten()(pool_2)

    # Channel 3 - third binary tile color
    input_3 = Input(shape=data_shape)
    conv_3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_3)
    drop_3 = Dropout(0.5)(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_3)
    flat_3 = Flatten()(pool_3)

    # Channel 4 - fourth binary tile color
    input_4 = Input(shape=data_shape)
    conv_4 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_4)
    drop_4 = Dropout(0.5)(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_4)
    flat_4 = Flatten()(pool_4)

    # Channel 5 - fifth binary tile color
    input_5 = Input(shape=data_shape)
    conv_5 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_5)
    drop_5 = Dropout(0.5)(conv_5)
    pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_5)
    flat_5 = Flatten()(pool_5)

    # Merge all of the different inputs into one tensor
    merged = concatenate(inputs=[flat_1, flat_2, flat_3, flat_4, flat_5])
    dense = Dense(10, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense)

    # Create the model with the separate 1-channel inputs and the single output
    model = Model(inputs=[input_1, input_2, input_3, input_4, input_5], outputs=output)

    # Use "mean squared error" as the loss function for regression and "mean absolute error"
    # as the metric to look at the network performance
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mae'])
    
    # Print the network structure to output for sanity check
    print(model.summary())
    # plot_model(model, show_shapes=True, to_file='network.png')

    return model


# Function to parse the board and split it into binary inputs
def split_input(board):
    inputs = np.zeros((len(tiles), len(board)), dtype='int_')
    for t in range(len(tiles)):
        for b in range(len(board)):
            if (board[b] == tiles[t]):
                inputs[t][b] = 1
    bg.print_data(inputs[1])
    return inputs


# data = bg.gen_random_data()
# print()
# bg.print_data(data)
# print()
# split_input(data)
# print()

build_model()


