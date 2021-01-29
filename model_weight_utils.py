import numpy as np

def gradients(prev_weights, new_weights):
    """
    gradients here should be added to the model, unlike the its conventional mathematical definition
    """
    gradients = []
    for i in range(len(prev_weights)):
        gradients.append(new_weights[i] - prev_weights[i])
    return gradients


def add_weights(w1, w2):
    if w1 == None:
        return w2
    res = []
    for i in range(len(w1)):
        res.append(w1[i] + w2[i])
    return res

def multiply_weights(w, num):
    res = []
    for i in range(len(w)):
        res.append(w[i] * num)
    return res

def avg_weights(my_weights, other_weights):
    if my_weights == None:
        return other_weights
    weights = [my_weights, other_weights]
    agg_weights = list()
    coeff = 0.5
    for weights_list_tuple in zip(*weights):
        agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[coeff, 1.-coeff]) for w in zip(*weights_list_tuple)]))
    
    return agg_weights
