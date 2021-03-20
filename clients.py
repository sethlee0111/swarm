import tensorflow.keras as keras
from tensorflow.keras import backend as K
import matplotlib
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
import tensorflow as tf
import data_process as dp
from data_process import get_sim_even, filter_data, get_kl_div, get_even_prob, JSD
import pickle
import models
import model_distance as md
from sklearn.metrics import f1_score
from model_weight_utils import *

def get_client_class(class_name):
    if class_name == 'momentum':
        client_class = MomentumClient
    elif class_name == 'momentum-wo-decay':
        client_class = MomentumWithoutDecayClient
    elif class_name == 'local':
        client_class = LocalClient
    elif class_name == 'greedy-sim':
        client_class = GreedySimClient
    elif class_name == 'greedy-no-sim':
        client_class = GreedyNoSimClient
    elif class_name == 'adv-greedy-sim':
        client_class = AdvancedGreedySimClient
    elif class_name == 'simple-momentum':
        client_class = SimpleMomentumClient
    elif class_name == 'jsd-momentum':
        client_class = JSDMomentumClient
    elif class_name == 'jsd-local-inc-momentum':
        client_class = JSDLocalIncMomentumClient
    elif class_name == 'simple-momentum-changing-local':
        client_class = SimpleMomentumChangingLocalClient
    elif class_name == 'simple-momentum-decay':
        client_class = SimpleMomentumAndDecayClient
    elif class_name == 'only-momentum':
        client_class = OnlyMomentumClient
    elif class_name == 'limited-momentum':
        client_class = LimitedMomentumClient
    elif class_name == 'limited-variant-momentum':
        client_class = LimitedVariantMomentumClient
    elif class_name == 'limited-variant-minmax-momentum':
        client_class = LimitedVariantMinMaxMomentumClient
    elif class_name == 'variant-momentum':
        client_class = VariantMomentumClient
    elif class_name == 'test-only-momentum':
        client_class = TestOnlyMomentumClient
    elif class_name == 'inf-momentum':
        client_class = InfMomentumClient
    elif class_name == 'new-momentum':
        client_class = NewMomentumClient
    elif class_name == 'kl-momentum':
        client_class = KLMomentumClient
    elif class_name == 'nodup-momentum':
        client_class = NoDupMomentumClient
    elif class_name == 'filtered-greedy':
        client_class = FilteredGreedySimClient
    elif class_name == 'kl-greedy-sim':
        client_class = KLGreedySimClient
    elif class_name == 'kl-greedy-only-sim':
        client_class = KLGreedyOnlySimClient
    else:
        return None
    return client_class

class DelegationClient:
    """client that runs local training and opportunistically a
    """
    def __init__(self, 
                _id,
                model_fn, 
                opt_fn,
                init_weights,
                x_train, 
                y_train,
                train_data_provider,
                test_data_provider,
                target_labels,
                compile_config, 
                train_config,
                hyperparams):
        """
        params
            model: function to get keras model
            init_weights: initial weights of the model
            x_train, y_train: training set
            compile_config: dict of params for compiling the model
            train_config: dict of params for training
            eval_config: dict of params for evaluation
        """
        self._id_num = _id
        self._model_fn = model_fn
        self._opt_fn = opt_fn
        self._weights = init_weights
        self._y_train_orig = y_train
        self.train_data_provider = train_data_provider
        self.test_data_provider = test_data_provider
        self._num_classes = test_data_provider.num_classes
        self._hyperparams = hyperparams
        self._evaluation_metrics = hyperparams['evaluation-metrics']
        self._similarity_threshold = hyperparams['similarity-threshold']

        ratio_per_label = 1./(len(target_labels))
        self._desired_data_dist = {}
        for l in target_labels:
            self._desired_data_dist[l] = ratio_per_label

        self._compile_config = compile_config
        self._train_config = train_config

        self.set_local_data(x_train, y_train)

        # print("client {} initialize".format(_id))
        # print("--desired_data: {}".format(self._desired_data_dist.keys()))
        # print("--local_data: {}".format(np.unique(y_train)))
    
    def set_local_data(self, x_train, y_train):
        self._x_train = x_train
        bc = np.bincount(y_train)
        ii = np.nonzero(bc)[0]
        self._local_data_dist = dict(zip(ii, bc[ii]/len(y_train)))
        self._local_data_count = dict(zip(ii, bc[ii]))
        self._y_train = keras.utils.to_categorical(y_train, self._num_classes)
    
    def replace_local_data(self, ratio, new_x_train, new_y_train_orig):
        """
        decrease the existing local set except the given ratio of data
        and add new train data on top of it
        """
        # if self._id_num % 20 == 0:
        #     print("replace for ratio {} in client{}".format(ratio, self._id_num))
        new_y_train = keras.utils.to_categorical(new_y_train_orig, self._num_classes)

        if ratio > 1:
            self._x_train = new_x_train
            self._y_train = new_y_train

        # shuffle existing data
        data_size = len(self._x_train)
        p = np.random.permutation(data_size)
        replace_size = (int)(data_size * (1-ratio))
        self._x_train = self._x_train[p][:replace_size]
        self._y_train = self._y_train[p][:replace_size]

        self._x_train = np.concatenate((self._x_train, new_x_train), axis=0)
        self._y_train = np.concatenate((self._y_train, new_y_train), axis=0)

    def local_data_dist_similarity(self, client):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._local_data_dist.keys() and i in client._local_data_dist.keys():
                overlap_ratio += min(self._local_data_dist[i], client._local_data_dist[i])
        return overlap_ratio

    def desired_data_dist_similarity(self, client):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._desired_data_dist.keys() and i in client._desired_data_dist.keys():
                overlap_ratio += min(self._desired_data_dist[i], client._desired_data_dist[i])
        return overlap_ratio

    def others_local_data_to_my_desired_data_similarity(self, other):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._desired_data_dist.keys() and i in other._local_data_dist.keys():
                overlap_ratio += min(self._desired_data_dist[i], other._local_data_dist[i])
        return overlap_ratio

    def min_d2l_data_similarity(self, other):
        desired_to_local_sim_me_to_other = self.others_local_data_to_my_desired_data_similarity(other)
        desired_to_local_sim_other_to_me = other.others_local_data_to_my_desired_data_similarity(self)

        min_dl_sim = min(desired_to_local_sim_me_to_other, desired_to_local_sim_other_to_me)
        return min_dl_sim

    def train(self, epoch):
        if epoch == 0:
            return {}

        model = self._get_model()
        
        self._train_config['epochs'] = epoch
        self._train_config['x'] = self._x_train
        self._train_config['y'] = self._y_train
        self._train_config['verbose'] = 0

        hist = model.fit(**self._train_config)
        self._weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        return hist

    def _train_with_others_data(self, other, epoch):
        if epoch == 0:
            return {}

        model = self._get_model()
        
        self._train_config['epochs'] = epoch

        # only pick the labels I want
        mask = np.zeros(other._y_train.shape[0], dtype=bool)
        for l in other._local_data_dist.keys():
            if l in self._local_data_dist.keys():
                one_hot = keras.utils.to_categorical(np.array([l]), 10)
                mask |= np.all(np.equal(other._y_train, one_hot), axis=1)

        if np.all(mask == False):
            return {}
        
        self._train_config['x'] = other._x_train
        self._train_config['y'] = other._y_train
        self._train_config['verbose'] = 0

        hist = model.fit(**self._train_config)
        self._weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        return hist

    def delegate(self, client, epoch, iteration):
        raise NotImplementedError("")

    def receive(self, client, coeff):
        """receive the model from the other client. Does not affect the other's model"""
        if coeff > 1 or coeff < 0:
            raise ValueError("coefficient is not in range 0 <= c <= 1: c:{}".format(coeff))
        
        weights = [self._weights, client._weights]
        agg_weights = list()
        for weights_list_tuple in zip(*weights):
            agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[coeff, 1.-coeff]) for w in zip(*weights_list_tuple)]))
        self._weights = copy.deepcopy(agg_weights)

    def eval(self):
        if self._evaluation_metrics == 'loss-and-accuracy':
            return self.eval_loss_and_accuracy()
        elif self._evaluation_metrics == 'f1-score-weighted':
            return self.eval_f1_score()
        elif self._evaluation_metrics == 'split-f1-score-weighted':
            return self.eval_split_f1_score()
        else:
            raise ValueError('evaluation metrics is invalid: {}'.format(self._evaluation_metrics))

    def eval_loss_and_accuracy(self):
        model = self._get_model()
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])                           
        hist = model.evaluate(xt, keras.utils.to_categorical(yt, self._num_classes), verbose=0)
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_f1_score(self, average='weighted'):
        model = self._get_model()
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])
        y_pred = np.argmax(model.predict(xt), axis = 1)
        hist = f1_score(yt, y_pred, average=average)
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_split_f1_score(self, average='weighted'):
        model = self._get_model()
        hist = {}
        for labels in self._hyperparams['split-test-labels']:
            xt, yt = self.test_data_provider.fetch(labels, self._hyperparams['test-data-per-label'])
            y_pred = np.argmax(model.predict(xt), axis = 1)
            if str(labels) not in hist:
                hist[str(labels)] = []
            hist[str(labels)].append(f1_score(yt, y_pred, average=average))
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_weights(self, weights):
        model = self._get_model_from_weights(weights)
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])                         
        hist = model.evaluate(xt, keras.utils.to_categorical(yt, self._num_classes), verbose=0)
        K.clear_session()
        del model
        return hist
    
    def _get_model(self):
        model = self._model_fn()
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model

    def _get_model_from_weights(self, weights):
        model = self._model_fn()
        model.set_weights(weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model
    
    def _get_model_w_lr(self, lr):
        model = self._model_fn()
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=lr)
        model.compile(**self._compile_config)
        return model

    def _get_dist_similarity(self, d1, d2):
        accum = 0.
        for k in d1.keys():
            if k in d2:
                accum += min(d1[k], d2[k])
        return accum
    
    def fit_to(self, other, epoch):
        model = self._get_model()
        self._train_config['epochs'] = 1
        self._train_config['x'] = other._x_train
        self._train_config['y'] = other._y_train
        self._train_config['verbose'] = 0
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_to_labels_in_my_goal(self, other, epoch):
        model = self._get_model()
        self._train_config['epochs'] = 1
        x, y = filter_data(other._x_train, other._y_train_orig, self._desired_data_dist.keys())
        self._train_config['x'] = x
        self._train_config['y'] = keras.utils.to_categorical(y, self._num_classes)
        if x.shape[0] < 1:
            raise ValueError("the filtered data size is 0!")
        self._train_config['verbose'] = 0
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_weights_to(self, weights, other, epoch):
        model = self._get_model_from_weights(weights)
        self._train_config['epochs'] = 1
        self._train_config['x'] = other._x_train
        self._train_config['y'] = other._y_train
        self._train_config['verbose'] = 0
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_w_lr_to(self, other, epoch, lr):
        model = self._get_model_w_lr(lr)
        self._train_config['epochs'] = 1
        self._train_config['x'] = other._x_train
        self._train_config['y'] = other._y_train
        self._train_config['verbose'] = 0
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        self._post_fit_w_lr_to(other)
        del model
        return weights
    
    def _post_fit_w_lr_to(self, other):
        pass

    def decide_delegation(self, other):
        return True

class SimularityDelegationClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        accum = 0.
        for k in self._desired_data_dist.keys():
            if k in other._local_data_dist:
                accum += min(self._desired_data_dist[k], other._local_data_dist[k])
        return accum

    def decide_delegation(self, other):
        return self.get_similarity(other) >= self._similarity_threshold

    # # @TODO this is a temporary function
    # def _post_fit_w_lr_to(self, other):
    #     x, y = other.train_data_provider.peek(other._local_data_count)
    #     other.set_local_data(x, y)

class KLSimularityDelegationClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        return self.get_kl_div(other._local_data_dist, self._desired_data_dist)

    def get_kl_div(self, dist_dict_1, dist_dict_2):
        kl_div = 0
        for k in dist_dict_1.keys():
            if k in dist_dict_2:
                kl_div += dist_dict_1[k] * np.log(dist_dict_1[k]/dist_dict_2[k])
            else:
                return np.inf
        return kl_div
    
    def decide_delegation(self, other):
        kl_div = self.get_similarity(other)
        if kl_div != np.nan and kl_div != np.inf:
            return kl_div <= 0.95
        return False

class AdvancedSimularityDelegationClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def get_dissimilarity(self, other):
        all_labels = list(range(self._num_classes))
        for k in self._desired_data_dist.keys():
            all_labels.remove(k)

        for k in other._local_data_dist.keys():
            if k in all_labels:
                nums += 1
        return nums/len(other._local_data_dist) < 0.5

    def decide_delegation(self, other):
        return self.get_similarity(other) >= self._similarity_threshold

class NoDupSimularityDelegationClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.last_five = []

    def decide_delegation(self, other):
        if self.get_similarity(other) >= self._similarity_threshold:
            if other._id_num not in self.last_five:
                self.last_five.append(other._id_num)
                self.last_five.pop(0)
                return True
            else:
                if len(self.last_five) > 0:
                    self.last_five.pop(0)
                return False

class LocalClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        # not in fact delegation at all
        for _ in range(iteration):
            self._weights = self.fit_to(self, 1)

    def decide_delegation(self, other):
        return True

class GreedyValidationClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        new_weights = self.fit_weights_to(self._weights, other, 1)
        new_weights = self.fit_weights_to(new_weights, self, 1)
        hist = self.eval_weights(new_weights)
        if self._last_hist[0] > hist[0]:
            self._weights = new_weights
            for _ in range(iteration-1):
                self._weights = self.fit_to(other, 1)
                self._weights = self.fit_to(self, 1)

class GreedyNoSimClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class GreedySimClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class KLGreedySimClient(KLSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class KLGreedyOnlySimClient(KLSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)

class FilteredGreedySimClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to_labels_in_my_goal(other, 1)
            self._weights = self.fit_to(self, 1)

class AdvancedGreedySimClient(AdvancedSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class AdvancedGreedyClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.lr_fac_min = 1
        self.init_weights = copy.deepcopy(self._weights)
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration):
        for _ in range(iteration):
            drift = md.l2_distance_w(self._weights, self.init_weights)
            xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
            lr_fac = np.exp(xx)/(np.exp(xx) + 1)
            self.lr_fac_min = min(self.lr_fac_min, lr_fac)
            lr = self.lr_fac_min * self._hyperparams['orig-lr']
            remote_weights = self.fit_w_lr_to(other,1, lr)
            remote_grads = gradients(self._weights, remote_weights)
            fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, list(other._local_data_dist.keys()))))
            remote_grads = multiply_weights(remote_grads, fac)
            self._weights = add_weights(self._weights, remote_grads)
            self._weights = self.fit_w_lr_to(self, 1, lr)

class SimpleMomentumClient(SimularityDelegationClient):
    """
    very similar to vanilla momentum.
    no overwriting subsets/supersets
    no decay
    @TODO not update with local, weight local
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_accum = None
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        if self._cache_accum == None:
            self._cache_accum = grads
        else:
            self._cache_accum = avg_weights(multiply_weights(grads, 0.5), multiply_weights(self._cache_accum, 0.5))

        for _ in range(iteration):
            self._weights = add_weights(self._weights, self._cache_accum)
            self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class SimpleMomentumChangingLocalClient(SimularityDelegationClient):
    """
    very similar to vanilla momentum.
    no overwriting subsets/supersets
    no decay
    @TODO not update with local, weight local
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_accum = None
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_fac = self._hyperparams['orig-lr']
        self.local_decay = 0.96

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']
        self.local_fac *= self.local_decay

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        if self._cache_accum == None:
            self._cache_accum = grads
        else:
            self._cache_accum = avg_weights(multiply_weights(grads, 0.5), multiply_weights(self._cache_accum, 0.5))

        for _ in range(iteration):
            self._weights = add_weights(self._weights, self._cache_accum)
            self._weights = self.fit_w_lr_to(self, 1, self.local_fac)

        K.clear_session()

class SimpleMomentumAndDecayClient(SimularityDelegationClient):
    """
    very similar to vanilla momentum.
    no overwriting subsets/supersets
    no decay
    @TODO not update with local, weight local
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_accum = None
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        if self._cache_accum == None:
            self._cache_accum = grads
        else:
            self._cache_accum = avg_weights(multiply_weights(grads, 0.5), multiply_weights(self._cache_accum, 0.5))

        for _ in range(iteration):
            self._weights = add_weights(self._weights, self._cache_accum)
            self._weights = self.fit_w_lr_to(self, 1, lr)

        # print("SimpleMomentumAndDecayClient lr: {}".format(lr))

        K.clear_session()

class SimpleWeightedMomentumClient(SimularityDelegationClient):
    """
    only a very simple book-keeping.
    no overwriting subsets/supersets
    no decay
    @TODO not update with local, weight local
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_accum = None
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        if self._cache_accum == None:
            self._cache_accum = grads
        else:
            self._cache_accum = avg_weights(multiply_weights(grads, 0.5), multiply_weights(self._cache_accum, 0.5))

        for _ in range(iteration):
            self._weights = add_weights(self._weights, self._cache_accum)
            self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class OnlyMomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        else:
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class TestOnlyMomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TEST 
                for c in found_supersets:
                    w1 = len(c) / (len(c) + len(other._local_data_dist.keys()))
                    w2 = len(other._local_data_dist.keys()) / (len(c) + len(other._local_data_dist.keys()))

                    self._cache_comb[c][1] = avg_weights(multiply_weights(self._cache_comb[c][1], w1),
                                                         multiply_weights(grads, w2))

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        else:
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class TestOnlyMomentumClient2(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        else:
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class InfMomentumClient_bkup(SimularityDelegationClient):
    """
    only update when we have "the full set"
    @TODO develop a client which updates and *reverts
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        # update cache
        self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        # check if we have all the labels
        keys = list(self._desired_data_dist.keys())
        for i in range(len(self._local_data_dist.keys())):
            keys.remove(i)
        for c in range(len(self._cache_comb)):
            for k in self._cache_comb[c][0]:
                try: 
                    keys.remove(k)
                except:
                    pass
    
        print("{} ... {}".format(other._local_data_dist.keys(), keys))

        # update if we have all the labels
        if len(keys) == 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            local_update = self.fit_w_lr_to(self, 1, lr)
            local_grads = gradients(self._weights, local_update)
            fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, self._local_data_dist.keys())))
            agg_g = add_weights(agg_g, multiply_weights(local_grads, fac))
            fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, multiply_weights(agg_g, 20))
            # flush all cache
            self._cache_comb = []

        K.clear_session()

class InfMomentumClient(SimularityDelegationClient):
    """
    only update when we have "the full set"
    @TODO develop a client which updates and *reverts
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        # update cache
        intersection = set(other._local_data_dist.keys()).intersection(set(self._desired_data_dist.keys())) 
        if len(intersection) > len(other._local_data_dist.keys())*0.5:
            self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        # check if we have all the labels
        keys = list(self._desired_data_dist.keys())
        for i in range(len(self._local_data_dist.keys())):
            keys.remove(i)
        for c in range(len(self._cache_comb)):
            for k in self._cache_comb[c][0]:
                try: 
                    keys.remove(k)
                except:
                    pass
    
        print("{} ... {}".format(other._local_data_dist.keys(), keys))

        # update if we have all the labels
        if len(keys) == 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            local_update = self.fit_w_lr_to(self, 1, lr)
            local_grads = gradients(self._weights, local_update)
            fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, self._local_data_dist.keys())))
            agg_g = add_weights(agg_g, multiply_weights(local_grads, fac))
            fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, multiply_weights(agg_g, 30))
            # flush all cache
            self._cache_comb = []

        K.clear_session()

class NewMomentumClient(SimularityDelegationClient):
    """
    only update when we have "the full set"
    @TODO develop a client which updates and *reverts
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        if not self.decide_delegation(other):
            return
        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        if len(found_subsets) != 0:
            agg_g = None
            for c in sorted(found_subsets, reverse=True):
                w1 = len(c) / (len(c) + len(other._local_data_dist.keys()))
                w2 = len(other._local_data_dist.keys()) / (len(c) + len(other._local_data_dist.keys()))
                agg_g = avg_weights(multiply_weights(self._cache_comb[c][1], w1),
                                                        multiply_weights(grads, w2))

            self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        else: # @TODO this is where I'm not too sure about
             for c in found_supersets:
                w1 = len(c) / (len(c) + len(other._local_data_dist.keys()))
                w2 = len(other._local_data_dist.keys()) / (len(c) + len(other._local_data_dist.keys()))

                self._cache_comb[c][1] = avg_weights(multiply_weights(self._cache_comb[c][1], w1),
                                                        multiply_weights(grads, w2))

        keys = self._desired_data_dist.keys()
        for c in range(len(self._cache_comb)):
            for k in self._cache_comb[c][0]:
                try: 
                    keys.remove(k)
                except:
                    pass

        if len(keys) == 0:


            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        else:
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class MomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
                # print(cc[0])
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class JSDMomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                fac = np.exp(-8*JSD(get_even_prob(cc[0]), self.desired_prob, self._num_classes))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                # print(cc[0])
            agg_g = multiply_weights(agg_g, 2*self._hyperparams['apply-rate']/(len(self._cache_comb)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class JSDLocalIncMomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self._cache_comb.append([set(self._local_data_dist.keys()), 0])
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            self._cache_comb.append([set(other._local_data_dist.keys()), grads])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        # update gradients for local
        new_weights = self.fit_w_lr_to(self, 1, lr)
        grads = gradients(self._weights, new_weights)
        for c in self._cache_comb:
            if c[0] == set(self._local_data_dist.keys()):
                c[1] = grads
                # c[2] = 0

        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                print(cc[0])
                fac = np.exp(-8*JSD(get_even_prob(cc[0]), self.desired_prob, self._num_classes))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                # print(cc[0])
            agg_g = multiply_weights(agg_g, 6*self._hyperparams['apply-rate']/(len(self._cache_comb)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)

        K.clear_session()

class LimitedMomentumClient(SimularityDelegationClient):
    """
    gradient goes stale after being applied to some
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # (label_set, gradient, staleness)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            # overwrite found subsets (e.g. [0,1,2] + [0,1] -> [0,1,2])
            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads, 0])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)
                    self._cache_comb[c][2] = 0 # reset staleness

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            remove_list = [] # list of stale gradients
            for cc in self._cache_comb:
                if cc[2] < 4:
                    fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                    agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                    fac_sum += fac
                    cc[2] += 1
                else:
                    remove_list.append(cc)
            for r in remove_list:
                self._cache_comb.remove(r)
            if len(self._cache_comb) < 1:
                K.clear_session()
                return
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class LimitedVariantMomentumClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # (label_set, gradient, staleness)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            # overwrite found subsets (e.g. [0,1,2] + [0,1] -> [0,1,2])
            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads, 0])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        # remove stale gradients
        remove_list = []
        for cc in self._cache_comb:
            if cc[2] > 3:
                remove_list.append(cc)
        for r in remove_list:
            self._cache_comb.remove(r)

        not_covered_labels = list(self._desired_data_dist.keys())
        for cc in self._cache_comb:
            for label in cc[0]:
                # print(label)
                if label in not_covered_labels:
                    not_covered_labels.remove(label)
        for l in self._local_data_dist.keys():
            if l in not_covered_labels:
                not_covered_labels.remove(l)
        # print(not_covered_labels)

        if len(not_covered_labels)/len(self._desired_data_dist) <= 0.2:
            agg_g = None
            fac_sum = 0
            remove_list = [] # list of stale gradients
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
                cc[2] += 1
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class LimitedVariantMinMaxMomentumClient(SimularityDelegationClient):
    """
    LimitedVariant + 
        we do not let the differenct btw. max & min number of times a label is applied above some threshold
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # (label_set, gradient, staleness)
        self._cache_comb.append([set(self._local_data_dist.keys()), 0, 0])
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        self.desired_prob = {}
        for l in self._desired_data_dist.keys():
            self.desired_prob[l] = 1./len(self._desired_data_dist.keys())
        self.labels_apply_nums = {}
        for l in self._desired_data_dist.keys():
            self.labels_apply_nums[l] = 0

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        # overwrite found subsets (e.g. [0,1,2] + [0,1] -> [0,1,2])
        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, 0])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        # remove stale gradients
        remove_list = []
        for cc in self._cache_comb:
            if cc[2] > 25:
                remove_list.append(cc)
        for r in remove_list:
            self._cache_comb.remove(r)

        not_covered_labels = list(self._desired_data_dist.keys())
        for cc in self._cache_comb:
            for label in cc[0]:
                # print(label)
                if label in not_covered_labels:
                    not_covered_labels.remove(label)
        for l in self._local_data_dist.keys():
            if l in not_covered_labels:
                not_covered_labels.remove(l)
        # print(not_covered_labels)

        # update gradients for local
        new_weights = self.fit_w_lr_to(self, 1, lr)
        grads = gradients(self._weights, new_weights)
        for c in self._cache_comb:
            if c[0] == set(self._local_data_dist.keys()):
                c[1] = grads
                c[2] = 0

        if len(not_covered_labels)/len(self._desired_data_dist) <= 0.2:
            agg_g = None
            remove_list = [] # list of stale gradients
            print("---------")
            for cc in self._cache_comb: # Only apply gradients that doesn't make minmax exceed given thres.
                after_labels_apply_nums = copy.deepcopy(self.labels_apply_nums)
                for l in cc[0]:
                    if l in list(self._desired_data_dist.keys()):
                        after_labels_apply_nums[l] += 1
                max_label = max(after_labels_apply_nums.values())
                min_label = min(after_labels_apply_nums.values())
                print(cc[0])
                if max_label - min_label < 25:
                    fac = np.exp(-8*JSD(get_even_prob(cc[0]), self.desired_prob, self._num_classes))
                    print(fac)
                    if fac <= 0:
                        continue
                    agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                    cc[2] += 1
                    self.labels_apply_nums = copy.deepcopy(after_labels_apply_nums)                    
                else:
                    print('not applied: {}'.format(self.labels_apply_nums))

            if agg_g == None:
                K.clear_session()
                return
            agg_g = multiply_weights(agg_g, 6*self._hyperparams['apply-rate']/(len(self._cache_comb)))
            # do training
            self._weights = add_weights(self._weights, agg_g)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class LimitedVariantMinMaxMomentumClientBKUP(SimularityDelegationClient):
    """
    LimitedVariant + 
        we do not let the differenct btw. max & min number of times a label is applied above some threshold
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # (label_set, gradient, staleness)
        self._cache_comb.append([set(self._local_data_dist.keys()), 0, 0])
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        self.desired_prob = {}
        for l in self._desired_data_dist.keys():
            self.desired_prob[l] = 1./len(self._desired_data_dist.keys())
        self.labels_apply_nums = {}
        for l in self._desired_data_dist.keys():
            self.labels_apply_nums[l] = 0

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)

        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        # overwrite found subsets (e.g. [0,1,2] + [0,1] -> [0,1,2])
        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, 0])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        # remove stale gradients
        remove_list = []
        for cc in self._cache_comb:
            if cc[2] > 10:
                remove_list.append(cc)
        for r in remove_list:
            self._cache_comb.remove(r)
        # update gradients for local
        new_weights = self.fit_w_lr_to(self, 1, lr)
        grads = gradients(self._weights, new_weights)
        for c in self._cache_comb:
            if c[0] == set(self._local_data_dist.keys()):
                c[1] = grads
                c[2] = 0

        agg_g = None
        remove_list = [] # list of stale gradients
        print("---------")
        not_covered_labels = list(self._desired_data_dist.keys())

        for cc in self._cache_comb: # Only apply gradients that doesn't make minmax exceed given thres.
            after_labels_apply_nums = copy.deepcopy(self.labels_apply_nums)
            for l in cc[0]:
                if l in list(self._desired_data_dist.keys()):
                    after_labels_apply_nums[l] += 1
            max_label = max(after_labels_apply_nums.values())
            min_label = min(after_labels_apply_nums.values())
            print(cc[0])
            if max_label - min_label < 25:
                fac = max(0.7-JSD(get_even_prob(cc[0]), self.desired_prob, self._num_classes), 0)
                print(fac)
                if fac <= 0:
                    continue
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                for l in cc[0]:
                    if l in not_covered_labels:
                        not_covered_labels.remove(l)
                cc[2] += 1
                self.labels_apply_nums = copy.deepcopy(after_labels_apply_nums)                    
            else:
                print('not applied: {}'.format(self.labels_apply_nums))

            if agg_g == None or len(not_covered_labels)/len(self._desired_data_dist) > 0.8:
                print('pass update, not covered labels: {}'.format(not_covered_labels))
                K.clear_session()
                return
            agg_g = multiply_weights(agg_g, 2*self._hyperparams['apply-rate']/(len(self._cache_comb)))
            # do training
            self._weights = add_weights(self._weights, agg_g)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()
class VariantMomentumClient(SimularityDelegationClient):
    """
    apply gradients only when they cover all the goal set
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        goal_set = list(self._desired_data_dist.keys())
        for cc in self._cache_comb:
            for label in cc[0]:
                # print(label)
                if label in goal_set:
                    goal_set.remove(label)
        # print(goal_set)

        if len(goal_set) == 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class KLMomentumClient(KLSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class KLMomentumWeightedLocalClient(KLSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            # weight the gradients from local
            local_update = self.fit_w_lr_to(self, 1, lr)
            grads = gradients(self._weights, new_weights)
            fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, self._local_data_dist.keys())))
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)

        K.clear_session()

class NoDupMomentumClient(NoDupSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class NoDupMomentumClient(NoDupSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class MomentumWithoutDecayClient(SimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)

    def delegate(self, other, epoch, iteration=2):
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        lr_fac = 1
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, 1, lr)
        grads = gradients(self._weights, new_weights)
        if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
            # update cache
            found_subsets = [] # smaller or equal
            found_supersets = [] # bigger
            for c in range(len(self._cache_comb)):
                if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                    found_subsets.append(c)
                elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                        len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                    found_supersets.append(c)

            if len(found_supersets) == 0:
                if len(found_subsets) != 0:
                    for c in sorted(found_subsets, reverse=True):
                        del self._cache_comb[c]
                self._cache_comb.append([set(other._local_data_dist.keys()), grads])

            else: # @TODO this is where I'm not too sure about
                for c in found_supersets:
                    self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        if len(self._cache_comb) > 0:
            agg_g = None
            fac_sum = 0
            for cc in self._cache_comb:
                fac = np.exp(-7 * (1-get_sim_even(self.desired_prob, cc[0])))
                agg_g = add_weights(agg_g, multiply_weights(cc[1], fac))
                fac_sum += fac
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate']/(len(self._cache_comb)*(fac_sum)))
            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                self._weights = self.fit_w_lr_to(self, 1, lr)
        else:
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()