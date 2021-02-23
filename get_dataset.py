from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from utils.har.sliding_window import sliding_window
import _pickle as cp

def get_mnist_dataset():
    # import dataset
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return x_train, y_train_orig, x_test, y_test_orig

def get_cifar_dataset():
    img_rows, img_cols = 32, 32
    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = tf.keras.datasets.cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train_orig = y_train_orig.reshape(-1)
    y_test_orig = y_test_orig.reshape(-1)

    return x_train, y_train_orig, x_test, y_test_orig

def get_opp_uci_dataset(filename, sliding_window_length, sliding_window_step):
    # from https://github.com/STRCWearlab/DeepConvLSTM

    with open(filename, 'rb') as f:
        data = cp.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    X_train, Y_train = opp_sliding_window(X_train, y_train, sliding_window_length, sliding_window_step)
    X_test, Y_test = opp_sliding_window(X_test, y_test, sliding_window_length, sliding_window_step) 

    return  np.expand_dims(X_train, axis=3), Y_train,  np.expand_dims(X_test, axis=3), Y_test

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)