import sys

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, load_model

import numpy as np
import matplotlib.pyplot as plt

import BoardGen as bg


# Include the empty tiles as an input?
# tiles = ['.', 'p', 'b', 'y', 'r', 'g']
tiles = ['p', 'b', 'y', 'r', 'g']

# DEFINES THE OLD MODEL
def build_model(channel_length=80, data_shape=(8, 10, 1), conv_filters=32):
    """

    The shape of the input data has to be a tuple:
    (<batch_size>, <rows>, <cols>, <channels>)

    For this network, the channels are split across different input images.
    So each individual input image should have the shape (<rows>, <cols>, 1).

    """

    # Channel 1 - first binary tile color
    input_1 = Input(shape=data_shape)
    conv_1 = Conv2D(filters=conv_filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_1)
    drop_1 = Dropout(0.5)(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_1)
    flat_1 = Flatten()(pool_1)

    # Channel 2 - second binary tile color
    input_2 = Input(shape=data_shape)
    conv_2 = Conv2D(filters=conv_filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_2)
    drop_2 = Dropout(0.5)(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_2)
    flat_2 = Flatten()(pool_2)

    # Channel 3 - third binary tile color
    input_3 = Input(shape=data_shape)
    conv_3 = Conv2D(filters=conv_filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_3)
    drop_3 = Dropout(0.5)(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_3)
    flat_3 = Flatten()(pool_3)

    # Channel 4 - fourth binary tile color
    input_4 = Input(shape=data_shape)
    conv_4 = Conv2D(filters=conv_filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_4)
    drop_4 = Dropout(0.5)(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop_4)
    flat_4 = Flatten()(pool_4)

    # Channel 5 - fifth binary tile color
    input_5 = Input(shape=data_shape)
    conv_5 = Conv2D(filters=conv_filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=data_shape)(input_5)
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
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy', 'mape'])
    
    # Print the network structure to output for sanity check
    print(model.summary())
    # plot_model(model, show_shapes=True, to_file='network.png')

    return model

# NOT USED IN NEW MODEL
# Function to parse a single board and split it into binary inputs
def split_input(board):
    inputs = np.zeros((len(tiles), len(board)), dtype='int_')
    for t in range(len(tiles)):
        for b in range(len(board)):
            if (board[b] == tiles[t]):
                inputs[t][b] = 1
    bg.print_data(inputs[1])
    return inputs

# NOT USED IN NEW MODEL
# Function to parse a dataset of boards and split them into binary inputs
def split_inputs(dataset, input_shape=(8, 10, 1)):
    
    # There is a separate list for each of the different tile colors
    separated_data = [[] for t in range(len(tiles))]

    # For each input board split it into the different binary boards
    for i in range(len(dataset)):
        for t in range(len(tiles)):
            binary_board = np.zeros(len(dataset[i]), dtype='int_')
            for b in range(len(dataset[i])):
                if (dataset[i][b] == tiles[t]):
                    binary_board[b] = 1
            separated_data[t].append(binary_board.reshape(input_shape))

    # Convert each list of binary boards to numpy array
    separated_data = [np.array(x) for x in separated_data]
    return separated_data


def define_model(filters=32, shape=(8, 10, 5)):
    """Defines a deep convolutional net that can be used for Superball.

    Arguments:
        filters (int): The number of convolutional filters to use.
        shape (tuple): The shape of each input board. Should be in the format
                       (<rows>, <cols>, <channels>), where each channel is
                       a binary projection of the board for one tile color.
    Returns:
        (tf.keras model): A trainable model of a convolutional net.
    """
    model = Sequential([
        Conv2D(filters=filters, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=shape),
        Dropout(0.5),
        #MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1) # Default activation is linear: y(x)=x
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    print(model.summary())
    return model

# THIS FUNCTION ASSUMES A DATASET LARGER THAN 1, WHICH COULD CAUSE PROBLEMS FOR ONLINE LEARNING
def split_into_channels(dataset):
    """A function to split each board into 5 binary color channels.

    Arguments:
        dataset (numpy array): A set of all of the boards in the dataset. The
                               shape of each input board is (80)
    Returns:
        (numpy array): A set of all boards in the dataset with an added
                       dimension (the number of channels). The shape of each
                       output board is (8, 10, 5).
    """
    new_dataset = []
    for i in range(len(dataset)):
        channels = np.zeros((8, 10, len(tiles)), dtype='int_')
        for t in range(len(tiles)):
            for b in range(len(dataset[i])):
                if (dataset[i][b] == tiles[t]):
                    channels[b//10][b%10][t] = 1
        new_dataset.append(channels)
    return np.array(new_dataset)


if __name__ == "__main__":

    # Load a random board dataset from BoardGen.py
    (train_data, train_labels), (test_data, test_labels) = bg.load_data()
    print(train_data.shape)

    separated_data = split_into_channels(train_data)
    print('shape', separated_data.shape)


    # Normalize the labels to values between -1 and 1
    # Uses a Gaussian distribution
    mean = train_labels.mean(axis=0)
    std = train_labels.std(axis=0)
    train_labels_in = (train_labels - mean) / std
    print('mean', mean, 'std', std)


    path = 'model.h5'
    if (len(sys.argv) == 2 and sys.argv[1] == 'load'):
        model = load_model(path)
    else:
        EPOCHS = 100
        model = define_model()
        model.fit(separated_data, train_labels_in, epochs=EPOCHS, batch_size=32)
        model.save(path)

    # Separate the test data into channels
    # separated_test_data = split_into_channels(test_data)
    # predictions = model.predict(separated_test_data).flatten()
    predictions = model.predict(separated_data).flatten()

    # Print the last 100 predicted values alongside the actual values
    for i in range(900, len(train_labels)):
        print((predictions[i]*std)+mean,':',train_labels[i])

    # The stuff after this line is from the old model
    exit()



# Split the data into 5 sets of binary boards
# separated_data = split_inputs(train_data)
# print('len of separated data:', len(separated_data))
# print('shape of separated_data[0]:', separated_data[0].shape)

# Normalize the labels to values between 0 and 1
# x_min = train_labels.min(axis=0)
# x_max = train_labels.max(axis=0)
# x_dif = x_max - x_min
# train_labels = (train_labels - x_min) / x_dif
# print('min', x_min)
# print('max', x_max)
# print(train_labels[0])

# EPOCHS = 100
# model = build_model()
# model.fit([x for x in separated_data], train_labels, epochs=EPOCHS, batch_size=2)

# separated_test_data = split_inputs(test_data)
# predictions = model.predict([x for x in separated_test_data]).flatten()
# predictions = [(predictions[i]*std)+mean for i in range(len(predictions))]
# predictions = [(predictions[i]*x_dif)+x_min for i in range(len(predictions))]

# print('mean:', mean)
# print('std:', std)
# print('predicted:', predictions)
# print('actual:', test_labels)




