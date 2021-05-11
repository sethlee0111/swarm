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
from sklearn.utils import shuffle

def get_client_class(class_name):
    if class_name == 'greedy':
        client_class = GreedyNoSimClient
    elif class_name == 'opportunistic':
        client_class = JSDGreedySimClient
    elif class_name == 'opportunistic-weighted':
        client_class = JSDWeightedGreedySimClient
    elif class_name == 'opportunistic (high thres.)':
        client_class = HighJSDGreedySimClient
    elif class_name == 'federated':
        client_class = FederatedGreedyNoSimClient
    elif class_name == 'federated (opportunistic)':
        client_class = FederatedJSDGreedyClient
    elif class_name == 'gradient replay':
        client_class = JSDGradientReplayClient
    elif class_name == 'gradient replay decay':
        client_class = JSDGradientReplayDecayClient
    elif class_name == 'gradient replay (high thres.)':
        client_class = HighJSDLocalIncStaleMomentumClient
    # 'cecay': Client-specific dECAY
    elif class_name == 'greedy-cecay':
        client_class = GreedyNoSimCecayClient
    elif class_name == 'opportunistic-cecay':
        client_class = JSDGreedySimCecayClient
    elif class_name == 'gradient replay cecay':
        client_class = JSDGradientReplayCecayClient
    elif class_name == 'greedy ':
        client_class = OnlyOtherGreedyClient
    elif class_name == 'oracle':
        client_class = OracleClient
    elif class_name == 'task-aware':
        client_class = TaskAwareClient
    elif class_name == 'compressed':
        client_class = CompressedTaskAwareClient
    elif class_name == 'compressed-v2':
        client_class = V2CompressedTaskAwareClient
    elif class_name == 'task-aware GR':
        client_class = TaskAwareGradientReplayClient
    ### CIFAR 100 versions
    elif class_name == 'oracle ':
        client_class = Cifar100OracleClient
    elif class_name == 'compressed ':
        client_class = CompressedCNNTaskAwareClient
    elif class_name == 'compressed-v2 ':
        client_class = V2CompressedCNNTaskAwareClient

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
        self.task_num = train_data_provider.task_num

        self.last_batch_num = {} # keeps the last batch num of the other client that this client was trained on
        self.total_num_batches = int(len(y_train) / hyperparams['batch-size'])
        if len(y_train) / hyperparams['batch-size'] - self.total_num_batches != 0:
            raise ValueError('batch-size has to divide local data size without remainders')

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

    def set_task(self, task):
        self.task = task

    def get_task(self):
        return self.task
    
    def set_local_data(self, x_train, y_train):
        bc = np.bincount(y_train)
        ii = np.nonzero(bc)[0]
        self._local_data_dist = dict(zip(ii, bc[ii]/len(y_train)))
        self._local_data_count = dict(zip(ii, bc[ii]))
        x_train, y_train = shuffle(x_train, y_train)
        self._x_train = x_train
        self._y_train_orig = y_train
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

    def get_samples(self, data_size=1):
        label_conf = {}
        for l in self._local_data_dist:
            label_conf[l] = data_size
        sample_data_provider = dp.DataProvider(self._x_train, self._y_train_orig, 0)
        return sample_data_provider.peek(label_conf)

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
        raise NotImplementedError('')

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

    def _get_compressed_model(self):
        model = self._model_fn(compressed_ver=1)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 30
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_v2_compressed_model(self):
        model = self._model_fn(compressed_ver=2)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 50
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[0] = weights[0][:, choice]
        weights[1] = weights[1][choice]
        weights[2] = weights[2][choice, :]

        COMPRESSED_LAYER_SIZE = 30
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_compressed_cnn_model(self):
        model = self._model_fn(compressed_ver=1)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 128
        a = np.arange(512)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_v2_compressed_cnn_model(self):
        model = self._model_fn(compressed_ver=2)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 64
        a = np.arange(512)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
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

    def get_batch_num(self, other):
        if other._id_num in self.last_batch_num:
            self.last_batch_num[other._id_num] = (self.last_batch_num[other._id_num] + 1) % self.total_num_batches
        else:
            self.last_batch_num[other._id_num] = 0
        return self.last_batch_num[other._id_num]

    def fit_to(self, other, epoch):
        """
        fit the model to others data for "epoch" epochs
        one epoch only corresponds to a single batch
        """
        model = self._get_model()
        
        model.fit(**self.get_train_config(other, epoch, self.get_batch_num(other)))
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
        model.fit(**self.get_train_config(other, epoch))
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_w_lr_to(self, other, epoch, lr):
        model = self._get_model_w_lr(lr)
        model.fit(**self.get_train_config(other, epoch, self.get_batch_num(other)))
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def get_train_config(self, client, steps, batch_num=None):
        tc = copy.deepcopy(self._train_config)
        tc['steps_per_epoch'] = steps
        tc['epochs'] = 1
        if batch_num == None:
            tc['x'] = client._x_train
            tc['y'] = client._y_train
        else:
            idx_start = batch_num * self._hyperparams['batch-size']
            if idx_start > client._x_train.shape[0]:
                raise ValueError('batch number is too large')
            idx_end = min(idx_start + self._hyperparams['batch-size'], client._x_train.shape[0])
            tc['x'] = client._x_train[idx_start:idx_end]
            tc['y'] = client._y_train[idx_start:idx_end]
        tc['verbose'] = 0
        return tc

    def decide_delegation(self, other):
        return True

    def is_federated(self):
        return False

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
        return np.exp(-8*get_kl_div(other._local_data_dist, self._desired_data_dist, self._num_classes))
    
    def decide_delegation(self, other):
        kl_div = self.get_similarity(other)
        if kl_div != np.nan and kl_div != np.inf:
            return kl_div <= 0.95
        return False

class JSDSimularityDelegationClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        # print("other: {}".format(other._local_data_dist.keys()))
        return JSD(other._local_data_dist, self._desired_data_dist, self._num_classes)
    
    def decide_delegation(self, other):
        jsd = self.get_similarity(other)
        if jsd != np.nan and jsd != np.inf:
            return jsd <= 0.5
        return False

class HighJSDSimularityDelegationClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        return JSD(other._local_data_dist, self._desired_data_dist, self._num_classes)
    
    def decide_delegation(self, other):
        jsd = self.get_similarity(other)
        if jsd != np.nan and jsd != np.inf:
            return jsd <= 0.4
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
            self._weights = self.fit_to(other, epoch)
            self._weights = self.fit_to(self, epoch)

class OracleClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other._id_num == 100 + self.task_num:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class Cifar100OracleClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        mammals = {'fox': 34, 'porcupine': 63, 'possum': 64, 'raccoon': 66, 'skunk': 75}
        flowers = {'orchid': 54, 'poppy': 63, 'rose': 70, 'sunflower': 82, 'tulip': 92}
        # print(other.get_task().split('-')[0])
        # print(other.get_task().split('-')[1])
        if other.get_task().split('-')[0] in mammals.keys() and other.get_task().split('-')[1] in flowers.keys():
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class OnlyOtherGreedyClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)

class TaskAwareClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_model()

    def get_logit_diff(self, other):
        np.set_printoptions(precision=3, suppress=True)
        model = self.fetch_model()
        extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
        other_x_samples, other_y_samples = other.get_samples()
        this_x_samples, this_y_samples = self.get_samples()
        other_x_samples = other_x_samples[np.argsort(other_y_samples)]
        other_y_samples = np.sort(other_y_samples)
        this_x_samples = this_x_samples[np.argsort(this_y_samples)]
        this_y_samples = np.sort(this_y_samples)
        if not np.any(other_y_samples == this_y_samples):
            raise ValueError('something wrong with sampling!')
        other_features = extractor(other_x_samples)
        this_features = extractor(this_x_samples)
        sums = []
        for i in range(len(other_y_samples)):
            diff = np.linalg.norm(np.array(other_features[-1][i]) - np.array(this_features[-1][i]))
            sums.append(diff)
            # print("LABEL: {} ---------------------".format(other_y_samples[i]))
            # import matplotlib.pyplot as plt
            # plt.imshow(this_x_samples[this_x_idx], cmap='gray')
            # plt.show()
            # import matplotlib.pyplot as plt
            # plt.imshow(other_x_samples[i], cmap='gray')
            # plt.show()
            # print(np.array(this_features[-1][i]))
            # print(np.array(other_features[-1][i]))
            # print(diff)
            # print(np.linalg.norm(np.array(other_features[-3][i]) - np.array(this_features[-3][this_x_idx])))
        sums.sort()
        # print(sums)
        # MEDIAN_PERCENTAGE = 70
        # lr_filtered_len = int((len(sums) * (1 - MEDIAN_PERCENTAGE / 100))/2)
        # print(lr_filtered_len)
        if len(sums) > 5:
            sums = sums[-5:]

        # compute consistency
        dev = 0
        for l in np.unique(other_y_samples):
            mask = other_y_samples == l
            other_features = extractor(other_x_samples[mask])
            for out in other_features[-1]:
                # print(out)
                dev += np.array(out[0]-out[1])
        dev /= len(np.unique(other_y_samples))
        # print(dev)
        
        return sum(sums) / len(sums)

    def decide_delegation(self, other):
        # print('self-diff')
        self_diff = self.get_logit_diff(self)
        # self_diff = 1
        # print('other-diff')
        other_diff = self.get_logit_diff(other)
        # print('task: {}, self: {}, other:{}, ratio: {}'.format(other.get_task(), self_diff, other_diff, other_diff/self_diff))
        ratio = other_diff/self_diff
        return ratio < self._hyperparams['task-threshold']

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class CompressedTaskAwareClient(TaskAwareClient):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_compressed_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.3

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class CompressedCNNTaskAwareClient(TaskAwareClient):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_compressed_cnn_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.3

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class V2CompressedTaskAwareClient(TaskAwareClient):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_v2_compressed_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.2

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(other, epoch, self._hyperparams['orig-lr'])


class V2CompressedCNNTaskAwareClient(TaskAwareClient):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_v2_compressed_cnn_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.2

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(other, epoch, self._hyperparams['orig-lr'])

class GreedyNoSimCecayClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.cecay_map = {}
        self.cecay = self._hyperparams['cecay']

    def delegate(self, other, epoch, iteration):
        if other._id_num in self.cecay_map:
            self.cecay_map[other._id_num] *= self.cecay
        else:
            self.cecay_map[other._id_num] = 1
        for _ in range(iteration):
            fac = self.cecay_map[other._id_num]
            update = self.fit_to(other, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac)
            self._weights = add_weights(self._weights, agg)

            fac = self.cecay_map[other._id_num]
            update = self.fit_to(self, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac)
            self._weights = add_weights(self._weights, agg)

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

class JSDGreedySimClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)
            self._weights = self.fit_to(self, epoch)

class JSDGreedySimCecayClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.cecay_map = {}
        self.cecay = self._hyperparams['cecay']

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        if other._id_num in self.cecay_map:
            self.cecay_map[other._id_num] *= self.cecay
        else:
            self.cecay_map[other._id_num] = 1
        for _ in range(iteration):
            fac = self.cecay_map[other._id_num]
            update = self.fit_to(other, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac)
            self._weights = add_weights(self._weights, agg)

            fac = self.cecay_map[other._id_num]
            update = self.fit_to(self, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac)
            self._weights = add_weights(self._weights, agg)

class JSDWeightedGreedySimClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            fac = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            update = self.fit_to(other, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac*10)
            self._weights = add_weights(self._weights, agg)
            # print("{}: {}".format(set(other._local_data_dist.keys()), fac))

            fac = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            update = self.fit_to(self, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac*10)
            self._weights = add_weights(self._weights, agg)

class HighJSDGreedySimClient(HighJSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)
            self._weights = self.fit_to(self, epoch)

class KLGreedyOnlySimClient(KLSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)

class FederatedGreedyNoSimClient(DelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.encountered_clients = {}

    def delegate(self, other, *args):
        self.encountered_clients[other._id_num] = other

    def federated_round(self, epoch):
        updates = []
        for k in self.encountered_clients:
            updates.append(self.fit_to(self.encountered_clients[k], epoch))
        agg_weights = list()
        for weights_list_tuple in zip(*updates):
            agg_weights.append(np.array([np.average(np.array(w), axis=0) for w in zip(*weights_list_tuple)]))
        self._weights = agg_weights

    def is_federated(self):
        return True

    def eval(self):
        return [0]
    
    def federated_eval(self):
        if self._evaluation_metrics == 'loss-and-accuracy':
            return self.eval_loss_and_accuracy()
        elif self._evaluation_metrics == 'f1-score-weighted':
            return self.eval_f1_score()
        elif self._evaluation_metrics == 'split-f1-score-weighted':
            return self.eval_split_f1_score()
        else:
            raise ValueError('evaluation metrics is invalid: {}'.format(self._evaluation_metrics))

class FederatedJSDGreedyClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.other_x_trains = None
        self.other_y_trains = None

    def delegate(self, other, *args):
        if not self.decide_delegation(other):
            return
        if self.other_x_trains == None or self.other_y_trains == None:
            self.other_x_trains = copy.deepcopy(other._x_train)
            self.other_y_trains = copy.deepcopy(other._y_train)
        else:
            self.other_x_trains = np.concatenate(self.other_x_trains, other._x_train)
            self.other_y_trains = np.concatenate(self.other_y_trains, other._y_train)

    def train(self, epochs):
        model = self._get_model()
        self._train_config['epochs'] = 1
        self._train_config['x'] = np.concatenate(self.other_x_trains, self._x_train)
        self._train_config['y'] = np.concatenate(self.other_y_trains, self._y_train)
        self._train_config['verbose'] = 0
        self._train_config['shuffle'] = True
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        self._weights = weights

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

class JSDGradientReplayClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (label_sets, gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_weight = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist)), self._desired_data_dist, self._num_classes))
        self.local_decay = 0.98
        self.local_apply_rate = 1

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, epoch, lr)
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
            weight = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, weight])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        stale_list = []
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[1], cc[2]))
                cc[2] *= 0.98 # @TODO add this to hyperparams
                if cc[2] < 0.001:
                    stale_list.append(cc)
            # remove stale gradients from the data structure
            for sl in stale_list:
                self._cache_comb.remove(sl)
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                new_weights = self.fit_w_lr_to(self, epoch, lr)
                grads = gradients(self._weights, new_weights)
                self._weights = add_weights(self._weights, multiply_weights(grads, self.local_weight * self.local_apply_rate * self._hyperparams['apply-rate']))
                self.local_apply_rate *= self.local_decay

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class JSDGradientReplayCecayClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (label_sets, gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_weight = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist)), self._desired_data_dist, self._num_classes))
        self.local_decay = 0.98
        self.local_apply_rate = 1

        self.cecay_map = {}
        self.cecay = self._hyperparams['cecay']

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return

        if other._id_num in self.cecay_map:
            self.cecay_map[other._id_num] *= self.cecay
        else:
            self.cecay_map[other._id_num] = 1

        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, epoch, lr)
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
            weight = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, self.cecay_map[other._id_num] * weight])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        stale_list = []
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[1], cc[2]))
                cc[2] *= 0.98 # @TODO add this to hyperparams
                if cc[2] < 0.001:
                    stale_list.append(cc)
            # remove stale gradients from the data structure
            for sl in stale_list:
                self._cache_comb.remove(sl)
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                new_weights = self.fit_w_lr_to(self, epoch, lr)
                grads = gradients(self._weights, new_weights)
                self._weights = add_weights(self._weights, multiply_weights(grads, self.local_weight * self.local_apply_rate * self._hyperparams['apply-rate']))
                self.local_apply_rate *= self.local_decay

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class TaskAwareGradientReplayClient(TaskAwareClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        self.local_decay = 0.98
        self.local_apply_rate = 1

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return

        weight = np.exp(-self.get_logit_diff(other))

        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, epoch, lr)
        grads = gradients(self._weights, new_weights)
        # update cache
        self._cache_comb.append([grads, weight])

        not_stale_list = []
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[0], cc[1]))
                cc[1] *= 0.3 # @TODO add this to hyperparams
                if not cc[1] < 0.005:
                    not_stale_list.append(cc)
            self._cache_comb = not_stale_list
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class HighJSDLocalIncStaleMomentumClient(HighJSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (label_sets, gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_weight = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist)), self._desired_data_dist, self._num_classes))

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, epoch, lr)
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
            weight = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, weight])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        stale_list = []
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[1], cc[2]))
                cc[2] *= 0.98 # @TODO add this to hyperparams
                if cc[2] < 0.001:
                    stale_list.append(cc)
            # remove stale gradients from the data structure
            for sl in stale_list:
                self._cache_comb.remove(sl)
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                new_weights = self.fit_w_lr_to(self, epoch, lr)
                grads = gradients(self._weights, new_weights)
                self._weights = add_weights(self._weights, multiply_weights(grads, self.local_weight * self._hyperparams['apply-rate']))

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()

class JSDLocalIncStaleMinMaxMomentumClient(JSDSimularityDelegationClient):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (label_sets, gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_weight = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist)), self._desired_data_dist, self._num_classes))

        # "MinMax"
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

        lr = self._hyperparams['orig-lr']

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
            weight = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, weight])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        # "MinMax"
        # Count the labels in the cache. If the label distribution of applied gradients doesn't satisfy our condition, then we defer model update
        for cc in self._cache_comb: 
            after_labels_apply_nums = copy.deepcopy(self.labels_apply_nums)
            for l in cc[0]:
                if l in list(self._desired_data_dist.keys()):
                    after_labels_apply_nums[l] += 1
        max_label = max(after_labels_apply_nums.values())
        min_label = min(after_labels_apply_nums.values())
        if max_label - min_label > 10:
            print('minmax condition not acheived')
            return

        stale_list = []
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[1], cc[2]))
                for l in cc[0]:
                    if l in list(self._desired_data_dist.keys()):
                        self.labels_apply_nums[l] += cc[2]
                cc[2] *= 0.98 # @TODO add this to hyperparams
                if cc[2] < 0.001:
                    stale_list.append(cc)
            # remove stale gradients from the data structure
            for sl in stale_list:
                self._cache_comb.remove(sl)
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                new_weights = self.fit_w_lr_to(self, 1, lr)
                grads = gradients(self._weights, new_weights)
                self._weights = add_weights(self._weights, multiply_weights(grads, self.local_weight * self._hyperparams['apply-rate']))
                for l in list(self._local_data_dist.keys()):
                    self.labels_apply_nums[l] += self.local_weight  # @TODO would this make the gradient apply stop at some point?

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)
        print(self.labels_apply_nums)
        K.clear_session()
