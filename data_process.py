import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import tensorflow.keras as keras
import copy

SIZE_X = 28
SIZE_Y = 28

# get X and Y data from a list of files
# returns: list of numpy arrays (num_samples_from_user, num_pixels)
def get_data(filenames):
    i = 0
    X = []
    Y = []
    users = []
    for fn in filenames:
        i += 1
        print("\r({}/{}) processing: {}".format(i, len(filenames), fn))
        with open(fn, "r") as f:
            data = f.read()
        parsed_data = json.loads(data)
        X.extend([np.array(parsed_data['user_data'][user]['x']) for user in parsed_data['users']])
        Y.extend([np.array(parsed_data['user_data'][user]['y']) for user in parsed_data['users']])
        users.extend(parsed_data['users'])
    return X, Y, users

# visualize the handwritten letters 
def visualize_writings(writing):
    map2d = []
    for i in range(0, len(writing), SIZE_Y):
        map2d.append(writing[i:i+SIZE_X])
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.array(map2d))
    fig.tight_layout()
    plt.show()

def get_even_prob(lst):
    dist = {}
    for l in lst:
        dist[l] = 1./len(lst)
    return dist
    
def get_kl_div(d1, d2, num_classes):
    kl_div = 0
    dist_dict_1 = copy.deepcopy(d1)
    dist_dict_2 = copy.deepcopy(d2)
    for l in range(num_classes):
        if l not in dist_dict_1:
            dist_dict_1[l] = 1e-09
        if l not in dist_dict_2:
            dist_dict_2[l] = 1e-09
    # print(dist_dict_1)
    # print(dist_dict_2)
    for k in dist_dict_1.keys():
        if k in dist_dict_2:
            kl_div += dist_dict_1[k] * np.log(dist_dict_1[k]/dist_dict_2[k])
        else:
            return np.inf
    return kl_div

def JSD(dist_dict_1, dist_dict_2, num_classes):
    d1 = copy.deepcopy(dist_dict_1)
    d2 = copy.deepcopy(dist_dict_2)
    for l in range(num_classes):
        if l not in d1:
            d1[l] = 1e-09
        if l not in d2:
            d2[l] = 1e-09
    dist_dict_m = {}
    for l in d1.keys():
        dist_dict_m[l] = (d1[l] + d2[l])/2
    return (get_kl_div(d1, dist_dict_m, num_classes) + \
            get_kl_div(d2, dist_dict_m, num_classes))/2

def get_sim_even(probs, labels):
        dist_dict = {}
        for l in labels:
            dist_dict[l] = 1./len(labels)
        tot = 0
        for k in probs.keys():
            if k in dist_dict:
                tot += min(probs[k], dist_dict[k])
        return tot
    
# parse data to global dataset and local sets for federated settings 
# every user is allocated to a global or local dataset, not being shared by two different sets
# this function only tries its best to fulfill requirements, it doesn't do error checking
# args: 
#      X: list of numpy arrays (num_samples_from_user, num_pixels)
#      num_global: minimum number of global data
#      num_local: minimum number of local data
# returns: X, Y for global, list of (X, Y)s for locals
def fl_parse(X, Y, num_clients, min_num_global, min_num_local):
    X_global = []
    Y_global = []
    local_data = []
    cnt = 0
    i = 0
    while i < len(X):
        X_global.append(X[i])
        Y_global.append(Y[i])
        cnt += X[i].shape[0]
        i += 1
        if cnt > min_num_global:
            break
            
    while len(local_data) < num_clients and i < len(X):
        X_local = []
        Y_local = []
        cnt = 0
        while cnt < min_num_local:
            X_local.append(X[i])
            Y_local.append(Y[i])
            cnt += X[i].shape[0]
            i += 1
        local_data.append((serialize_data(X_local), serialize_data(Y_local)))
        
    return serialize_data(X_global), serialize_data(Y_global), local_data

# split training set with given size and number
# args:
#      size: number of the data in each training set
#      x_train: numpy array of shape (num_samples, num_dimensions)
# returns: list of numpy array for X, Y
def split_training_set(size, number, x_train, y_train):
    x_train_list = np.split(x_train, x_train.shape[0] / size)[:number]  # +1 cuz the last array will contain everything till the end
    y_train_list = np.split(y_train, y_train.shape[0] / size)[:number]
    y_train_list = [keras.utils.to_categorical(y, len(np.unique(y_train))) for y in y_train_list]
    return x_train_list, y_train_list

def split_training_set_by_number(number, x_train, y_train):
    x_train_list = np.split(x_train, number)

def split_training_set_unbalanced(start_size, diff, number, x_train, y_train):
    num_shards = int(number * (number + 1) / 2)
    x_train_shards = np.split(x_train, x_train.shape[0] / diff)[:num_shards]
    y_train_shards = np.split(y_train, y_train.shape[0] / diff)[:num_shards]
    
    x_train_list = []
    y_train_list = []
    for i in range(number):
        if len(x_train_shards[:i+1]) != i+1:
            raise ValueError('train dataset not enough to construct given number of training set')
        x_train_list.append(np.concatenate(x_train_shards[:i+1], axis=0))
        x_train_shards = x_train_shards[i+1:]
        y_train_list.append(np.concatenate(y_train_shards[:i+1], axis=0))
        y_train_shards = y_train_shards[i+1:]
        
    y_train_list = [keras.utils.to_categorical(y, len(np.unique(y_train))) for y in y_train_list]
    return x_train_list, y_train_list

def filter_data_by_labels(x_train, y_train, labels, size=-1, noise=0):
    """
    return only the data with corresponding labels with noise
    note that the resulting size could be different from the parameter size.
    This is to ensure the number of data for each labels are exactly equal 
    """
    num_labels = len(np.unique(y_train))
    num_noise_labels = (num_labels - len(labels))
    num_true_labels = num_labels - num_noise_labels

    if num_noise_labels != 0:
        noise_size_per_label = (int)(size * noise / num_noise_labels)
    else:
        noise_size_per_label = 0
    true_size_per_label = (size - noise_size_per_label * num_noise_labels) / num_true_labels
    label_conf = dict()
    for i in np.unique(y_train):
        if i in labels:
            label_conf[i] = true_size_per_label
        else:
            label_conf[i] = noise_size_per_label

    print("filter by labels")
    print(label_conf)
    
    return filter_data_by_labels_with_numbers(x_train, y_train, label_conf)
    
def filter_data_by_labels_with_numbers(x_train, y_train, nums):
    """
    nums: a dict that specifies the number of data points per labels
    """
    if type(nums) != type({}):
        raise TypeError("nums has to be a dict type, not {}".format(type(nums)))
    
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
    
    total_data_size = len(y_train)
    
    mask = np.zeros(y_train.shape, dtype=bool)
    
    for l in nums.keys():
        new_mask = (y_train == l)
        cnt = 0
        for i in range(total_data_size):
            if new_mask[i]:
                if cnt >= nums[l]:
                    break
                cnt += 1

        mask |= np.append(new_mask[:i], np.zeros(total_data_size-i, dtype=bool))
        
    return x_train[mask], y_train[mask]

def filter_data(x_train, y_train, labels):
    """
    labels: a list of labels that'll be included in the returning training set
    """
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
    
    mask = np.zeros(y_train.shape, dtype=bool)
    
    for l in labels:
        label_mask = (y_train == l)
        mask |= label_mask
        
    return x_train[mask], y_train[mask]

class StableTestDataProvider():
    """
    when size_per_label = 0, use all the data for the test set
    """
    def __init__(self, x_test, y_test, size_per_label=0):
        self.x_test_set_by_labels = []
        self.y_test_set_by_labels = []
        if size_per_label != 0:
            for l in np.unique(y_test):
                p = np.random.permutation(size_per_label)
                self.x_test_set_by_labels.append(x_test[y_test == l][:size_per_label][p])
                self.y_test_set_by_labels.append(y_test[y_test == l][:size_per_label][p])
        else:
            unique, counts = np.unique(y_test, return_counts=True)
            count_dict = dict(zip(unique, counts))
            for l in np.unique(y_test):
                p = np.random.permutation(count_dict[l])
                self.x_test_set_by_labels.append(x_test[y_test == l][p])
                self.y_test_set_by_labels.append(y_test[y_test == l][p])

        self.num_classes = len(np.unique(y_test))

    def fetch(self, labels, size_per_label=0):
        if size_per_label != 0:
            xt = np.concatenate([self.x_test_set_by_labels[i][:size_per_label] for i in range(len(self.x_test_set_by_labels)) if i in labels], axis=0)
            yt = np.concatenate([self.y_test_set_by_labels[i][:size_per_label] for i in range(len(self.y_test_set_by_labels)) if i in labels])
        else:
            xt = np.concatenate([self.x_test_set_by_labels[i] for i in range(len(self.x_test_set_by_labels)) if i in labels], axis=0)
            yt = np.concatenate([self.y_test_set_by_labels[i] for i in range(len(self.y_test_set_by_labels)) if i in labels])
        return xt, yt

class DataProvider():
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.total_data_size = len(self.y_train)
        self.mask_unused = np.ones(self.total_data_size, dtype=bool)

        self.mask_per_label = []
        for l in np.unique(self.y_train):
            self.mask_per_label.append(self.y_train == l)
        
        # p = np.random.permutation(len(x_train))
        # self.x_train = x_train[p]
        # self.y_train = y_train[p]
    
    def fetch(self, labels):
        raise NotImplementedError('the change in peek() function made this function obsolete')
        # @WARNING: DO NOT use this method
        label_mask = np.zeros(self.total_data_size, dtype=bool)
        total_output_size = 0
        for l in labels.keys():
            total_output_size += labels[l]
            cnt = 0
            for i in range(self.total_data_size):
                if self.mask_unused[i] and self.mask_per_label[l][i]:
                    if cnt >= labels[l]:
                        break
                    cnt += 1
            label_mask |= np.append(self.mask_unused[:i] & self.mask_per_label[l][:i], np.zeros(self.total_data_size-i, dtype=bool))

        data_filter = label_mask
        x_filtered = self.x_train[data_filter]
        y_filtered = self.y_train[data_filter]

        self.mask_unused &= ~(data_filter)

        if x_filtered.shape[0] != total_output_size or y_filtered.shape[0] != total_output_size:
            raise ValueError("Dataset depleted. x: {}, y: {}".format(x_filtered.shape[0], y_filtered.shape[0]))
        
        return x_filtered, y_filtered
    
    def peek(self, labels):
        p = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[p]
        self.y_train = self.y_train[p]

        data_filter = np.zeros(self.total_data_size, dtype=bool)
        label_mask = np.zeros(self.total_data_size, dtype=bool)
        total_output_size = 0
        for l in labels.keys():
            total_output_size += labels[l]
            cnt = 0
            for i in range(self.total_data_size):
                if self.mask_unused[i] and self.y_train[i] == l:
                    label_mask[i] |= 1
                    cnt += 1
                    if cnt >= labels[l]:
                        break
            data_filter |= label_mask
            # label_mask |= np.append(self.mask_unused[:i] & self.mask_per_label[l][:i], np.zeros(self.total_data_size-i, dtype=bool))

        # data_filter = label_mask
        x_filtered = self.x_train[data_filter]
        y_filtered = self.y_train[data_filter]

        if x_filtered.shape[0] != total_output_size or y_filtered.shape[0] != total_output_size:
            raise ValueError("Dataset depleted. x: {}, y: {}".format(x_filtered.shape[0], y_filtered.shape[0]))
        
        return x_filtered, y_filtered

    def peek_orig(self, labels):
        label_mask = np.zeros(self.total_data_size, dtype=bool)
        total_output_size = 0
        for l in labels.keys():
            total_output_size += labels[l]
            cnt = 0
            for i in range(self.total_data_size):
                if self.mask_unused[i] and self.mask_per_label[l][i]:
                    if cnt >= labels[l]:
                        break
                    cnt += 1
            label_mask |= np.append(self.mask_unused[:i] & self.mask_per_label[l][:i], np.zeros(self.total_data_size-i, dtype=bool))

        data_filter = label_mask
        x_filtered = self.x_train[data_filter]
        y_filtered = self.y_train[data_filter]

        if x_filtered.shape[0] != total_output_size or y_filtered.shape[0] != total_output_size:
            raise ValueError("Dataset depleted. x: {}, y: {}".format(x_filtered.shape[0], y_filtered.shape[0]))
        
        return x_filtered, y_filtered

    def peek_categorical(self, labels, num_classes):
        x, y = self.peek(labels)
        return x, keras.utils.to_categorical(y, num_classes)

    def get_random(self, size):
        p = np.random.permutation(len(self.x_train))
        return self.x_train[p][:size], self.y_train[p][:size]

class TrainDataProvider():
    def __init__(self, x_test, y_test, size):
        """
        size: size of test set per label
        """
        # testsets per label
        self.x_test_list = []
        self.y_test_list = []
        
        for n in np.unique(y_test):
            xt, yt = filter_data_by_labels_with_numbers(x_test,
                                                        y_test,
                                                        {n:size})
            self.x_test_list.append(xt)
            self.y_test_list.append(yt)
    
    def fetch(self, labels):
        x = self.x_test_list[labels[0]]
        y = self.y_test_list[labels[0]]
        x_test = np.append(x, [self.x_test_list[n] for n in labels[1:]])
        y_test = np.append(y, [self.y_test_list[n] for n in labels[1:]])
        return x_test, y_test
    
# change list of numpy arrays (num_samples_from_user, num_pixels) to
# list of numpy arrays (num_pixels)
# in other words, erase user info and just serialize all the data
def serialize_data(X):
    res = []
    for x in X:
        res.extend(list(x))
    return np.array(res)

def get_train_data_from_filename(n):
    return "all_data_" + str(n) + "_niid_0_keep_10_train_9.json"

def get_test_data_from_filename(n):
    return "all_data_" + str(n) + "_niid_0_keep_10_test_9.json"
