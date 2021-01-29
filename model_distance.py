import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow.keras import backend as K
import scipy

def l1_distance(model1, model2):
    w = zip(model1.get_weights(), model2.get_weights())
    lw = list(w)
    sums = 0
    i = 0
    for ww in lw:
        i += 1
        sums += np.average(np.absolute(ww[0] - ww[1]))
    return sums

def l2_distance(model1, model2):
    if model1.count_params() != model2.count_params():
        raise ValueError("two models have different number of parameters")
    lw = list(zip(model1.get_weights(), model2.get_weights()))
    sums = []
    for ww in lw:
        norm = LA.norm(ww[0] - ww[1])
        sums.append(norm)
    return LA.norm(np.array(sums))

def l2_distance_w(model1, model2):
    lw = list(zip(model1, model2))
    sums = []
    for ww in lw:
        norm = LA.norm(ww[0] - ww[1])
        sums.append(norm)
    return LA.norm(np.array(sums))

def diag_fisher(model, data):
    """
    This function assumes that the last layer of the model has softmax activation
    """
    if model.layers[-1].activation.__name__ != 'softmax':
        from_logits = False
    if model.layers[-1].activation.__name__ != 'linear':
        from_logits = True
    else:
        raise InputError("The last layer has to have softmax or linear activation")
    
    xs = tf.Variable(data)
    with tf.GradientTape() as tape:
        y = model(xs)
        max_vals = tf.reduce_max(y, axis=1, keepdims=True)
        cond = tf.equal(y, max_vals)
        y_pred = tf.where(cond, tf.ones_like(y), tf.zeros_like(y))
        cc = K.categorical_crossentropy(y_pred, y, from_logits=False)
    tape_grad = tape.gradient(cc, model.trainable_variables)
    
    sess = K.get_session()
    sess.run(tf.variables_initializer([xs]))
    grads = sess.run(tape_grad)
    fisher = [g**2 for g in grads]
    fisher_flatten = np.concatenate([np.reshape(f, (-1)) for f in fisher]).reshape(-1)
    return fisher_flatten

def get_frechet_distance(tr1, tr2):
    """
    assume that m1 and m2 are already diagonals of some matrices
    """
    # normalize to have unit trace
    F1 = tr1 / np.linalg.norm(tr1)
    F2 = tr2 / np.linalg.norm(tr2)
    
    return (1/2) * sum((F1 + F2 - 2*np.sqrt(F1*F2)))

def get_fisher_distance(model1, model2, x_train):
    f1 = diag_fisher(model1, x_train)
    f2 = diag_fisher(model2, x_train)
    
    F1 = f1 / np.linalg.norm(f1)
    F2 = f2 / np.linalg.norm(f2)
    
    return np.linalg.norm(F1-F2)

def get_overlapping_top_fishers(model1, model2, x_train, top_num):
    f1 = diag_fisher(model1, x_train)
    f2 = diag_fisher(model2, x_train)
    
    tops1 = f1.argsort()[-top_num:]
    tops2 = f2.argsort()[-top_num:]
    
#     return np.intersect1d(tops1, tops2).size / top_num
    return get_intersect_params_coeff(tops1, tops2)
    
def get_intersect_params_coeff(tops1, tops2):
    itst, itst_ind_1, itst_ind_2 = np.intersect1d(tops1, tops2, return_indices=True)
    ratio = np.intersect1d(tops1, tops2).size / tops1.size
    itst_ind_1.sort()
    itst_ind_2.sort()
    corr, _ = scipy.stats.kendalltau(tops1[itst_ind_1], tops2[itst_ind_2])
    if ratio == np.nan or corr == np.nan:
        raise ValueError("ratio: {} or corr: {} is nan".format(ratio, corr))
    return (ratio + scipy.special.expit(corr, nan_policy='raise') * (1+np.exp(-1)))/2
    
    