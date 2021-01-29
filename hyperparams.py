import tensorflow.keras as keras

OPTIMIZER = keras.optimizers.SGD
LEARNING_RATE = 0.1
LOSS_FUNC = 'mean_squared_error'
METRICS = ['accuracy']
MAX_STEPS = 1000

BATCH_SIZE = 100
SIGMA = 5

ORIG_LR = 0.001
KAPPA = 14
OFFSET = 0.96
APPLY_RATE = 1