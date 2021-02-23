import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

# hyperparams for uci dataset
NUM_FILTERS = 64
FILTER_SIZE = 5
SLIDING_WINDOW_LENGTH = 24
NB_SENSOR_CHANNELS = 113
NUM_UNITS_LSTM = 128
NUM_CLASSES = 18

def get_2nn_mnist_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_2nn_cifar_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_cnn_mnist_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    return model

def get_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_big_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def get_better_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def get_deep_conv_lstm_model(num_filters=NUM_FILTERS,
                             filter_size=FILTER_SIZE,
                             sliding_window_length=SLIDING_WINDOW_LENGTH,
                             nb_sensor_channels=NB_SENSOR_CHANNELS,
                             num_units_lstm=NUM_UNITS_LSTM,
                             num_classes=NUM_CLASSES):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu',
                                  input_shape=(sliding_window_length, nb_sensor_channels, 1)))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    shape = model.layers[-1].output_shape
    model.add(keras.layers.Reshape((shape[1], shape[3] * shape[2])))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh', return_sequences=True)) # [batch, timesteps, features]
    model.add(keras.layers.Dropout(0.5, seed=123))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh'))
    model.add(keras.layers.Dropout(0.5, seed=124))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model