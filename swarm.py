import numpy as np
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras import backend as K
import copy
import pickle
import data_process as dp
import datetime
import logging
import numpy as np
from tqdm import tqdm

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class Swarm():
    def __init__(self, model_fn, opt_fn,
                 client_class,
                 num_clients, 
                 pretrain_model_weight,
                 x_train, y_train, 
                 test_data_provider,
                 num_label_per_client,
                 num_req_label_per_client,
                 num_data_per_label_in_client,
                 enc_exp_config, hyperparams, from_swarm=None):
        """
        enc_exp_config: dictionary for configuring encounter data based experiment
        [keys]
            data_file_name: pickle filename for pandas dataframe
            send_duration: how long does it take to send/receive the model?
            delegation_duration: how long does it take to run a single delegation
            max_delegations: maximum delegation rounds
        """
        self.test_data_provider = test_data_provider
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}
        self.hyperparams = hyperparams

        self.train_data_provider = dp.DataProvider(x_train, y_train)
        self.num_data_per_label_in_client = num_data_per_label_in_client

        num_unknown_label_per_client = max(num_req_label_per_client - num_label_per_client, 0)

        if from_swarm == None:
            self._clients = []
            # prepare data for quadrants
            # if enc_exp_config['mobility_model'] == 'levy_walk':
            try:
                self.local_data_per_quad = enc_exp_config['local_data_per_quad']
            except:
                self.local_data_per_quad = None
            # elif enc_exp_config['mobility_model'] == 'sigcomm2009':
            #     pass
            # else:
            #     raise ValueError('invalid mobility model: {}'.format(enc_exp_config['mobility_model']))
            
            # for i in range(9):
            #     candidates = np.arange(0,10)
            #     np.random.shuffle(candidates)
            #     self.local_data_per_quad.append(candidates[:num_label_per_client])

            for i in range(num_clients):
                # @TODO check if data is sufficient to go through all the simulation
                # prepare data

                # num_starts = np.arange(0, num_classes - num_label_per_client)
                # np.random.shuffle(num_starts)
                # local_data_labels = np.arange(num_starts[0], num_starts[0]+5)
                # if enc_exp_config['mobility_model'] == 'levy_walk':
                if self.local_data_per_quad is not None:
                    local_data_labels = self.local_data_per_quad[(int)(i / (int)(num_clients/9))]
                else:
                    local_candidates = np.array(np.arange(test_data_provider.num_classes))
                    np.random.shuffle(local_candidates)
                    local_data_labels = local_candidates[:num_label_per_client]
                # elif enc_exp_config['mobility_model'] == 'sigcomm2009':
                    

                # put random labels for required data
                target_labels_not_in_train_set = np.setdiff1d(np.arange(test_data_provider.num_classes),
                                                                np.array(local_data_labels))
                np.random.shuffle(target_labels_not_in_train_set)
                target_labels = np.concatenate((target_labels_not_in_train_set[:num_unknown_label_per_client],
                                                np.array(local_data_labels)))
                logging.info('client {} target: {}'.format(i, target_labels))
                tmp = [num_data_per_label_in_client] * len(local_data_labels)
                # x_train_client, y_train_client = self.train_data_provider.fetch(dict(zip(local_data_labels, tmp)))
                x_train_client, y_train_client = dp.filter_data_by_labels_with_numbers(x_train, y_train, dict(zip(local_data_labels, tmp)))
                self._clients.append(client_class(i,
                                    model_fn,
                                    opt_fn,
                                    copy.deepcopy(pretrain_model_weight),
                                    x_train_client,
                                    y_train_client,
                                    self.train_data_provider,
                                    self.test_data_provider,
                                    target_labels,  # assume that required d.d == client d.d.
                                    compile_config,
                                    train_config,
                                    hyperparams))

                # central server for classical federated learning simulation
                if i == 0:
                    self.central_server = client_class(1000,
                                                model_fn,
                                                opt_fn,
                                                copy.deepcopy(pretrain_model_weight),
                                                x_train_client,
                                                y_train_client,
                                                self.train_data_provider,
                                                self.test_data_provider,
                                                target_labels,  
                                                compile_config,
                                                train_config,
                                                hyperparams)
        else:
            self._clients = []
            for i in range(num_clients):
                self._clients.append(client_class(i,
                                    model_fn,
                                    opt_fn,
                                    copy.deepcopy(pretrain_model_weight),
                                    from_swarm._clients[i]._x_train,
                                    from_swarm._clients[i]._y_train_orig,
                                    from_swarm.train_data_provider,
                                    from_swarm.test_data_provider,
                                    list(from_swarm._clients[i]._desired_data_dist.keys()),  # assume that required d.d == client d.d.
                                    compile_config,
                                    train_config,
                                    hyperparams))
                
                # central server for classical federated learning simulation
                if i == 0:
                    self.central_server = client_class(-1,
                                                model_fn,
                                                opt_fn,
                                                copy.deepcopy(pretrain_model_weight),
                                                from_swarm._clients[i]._x_train,
                                                from_swarm._clients[i]._y_train_orig,
                                                from_swarm.train_data_provider,
                                                from_swarm.test_data_provider,
                                                list(from_swarm._clients[i]._desired_data_dist.keys()), 
                                                compile_config,
                                                train_config,
                                                hyperparams)
        
        self.hist = {} # history per client over time

        self.hist['clients'] = {}
        for i in range(num_clients):
            self.hist['clients'][i] = []

        self.hist['clients_unknown'] = {}
        for i in range(num_clients):
            self.hist['clients_unknown'][i] = []

        self.hist['clients_local'] = {}
        for i in range(num_clients):
            self.hist['clients_local'][i] = []
            
        self._config = enc_exp_config
        self.delegation_time = 2*self._config['send_duration'] + self._config['delegation_duration']
        self.enc_df = pd.read_pickle(self._config['data_file_name'])
        self.total_number_of_rows = self.enc_df.shape[0]

        self.hist['time_steps'] = [0]
        self.hist['loss_max'] = []
        self.hist['loss_min'] = []
        self.hist['total_delegations'] = 0
        self.hist['total_fr'] = 0

    def _evaluate_all(self):
        #  run one local updates each first
        for i in range(len(self._clients)):
            hist = self._clients[i].eval()
            self.hist['clients'][i].append((0, hist, [])) # assume clients all start from the same init
            self.hist['loss_max'].append(hist[0])
            self.hist['loss_min'].append(hist[0])

    def _initialize_last_times(self):
        self.last_end_time = {}
        for i in range(len(self._clients)):
            self.last_end_time[i] = 0
        self.last_data_update_time = {}
        for i in range(len(self._clients)):
            self.last_data_update_time[i] = 0

    def run(self):
        # stores the end time of the last encounter
        # this is to prevent one client exchanging with more than two
        # at the same time
        self._evaluate_all()
        self._initialize_last_times()

        print("Start running simulation with {} indices".format(self.total_number_of_rows))
        start_time = datetime.datetime.now()
        # iterate encounters
        cur_t = 0 # current time
        for index, row in self.enc_df.iterrows():
            cur_t = row[TIME_START]
            end_t = row[TIME_END]
            t_left = end_t - cur_t # time left
            # only pairs of clients can exchange in a place
            c1_idx = (int)(row[CLIENT1])
            c2_idx = (int)(row[CLIENT2])
            if c1_idx >= len(self._clients) or c2_idx >= len(self._clients):
                continue
            c1 = self._clients[c1_idx]
            c2 = self._clients[c2_idx]
            if self.last_end_time[c1_idx] > cur_t or self.last_end_time[c2_idx] > cur_t:
                continue  # client already occupied

            # assume both clients are fully occupied for delegation(so side-delegation in one time period)
            c1_delegate_to_c2 = c1.decide_delegation(c2)
            c2_delegate_to_c1 = c2.decide_delegation(c1)
            if c1_delegate_to_c2 or c2_delegate_to_c1:
                num_delegations = (int)(t_left/self.delegation_time)
                if num_delegations < 1:
                    continue
                delegations = min(num_delegations, self._config['max_delegations'])
                self.last_end_time[c1_idx] = cur_t + self.delegation_time * delegations
                self.last_end_time[c2_idx] = cur_t + self.delegation_time * delegations

            if c1_delegate_to_c2:
                # model delegation
                c1.delegate(c2, 1, delegations)
                self.hist['total_delegations'] += 1
                hist = c1.eval()
                self.hist['clients'][c1_idx].append((self.last_end_time[c1_idx], hist, list(c2._local_data_dist.keys())))
                # self.hist['loss_max'].append(max(self.hist['loss_max'][-1], hist[1])) # @TODO this won't work with eval_f1_score
                # self.hist['loss_min'].append(min(self.hist['loss_min'][-1], hist[1]))
            if c2_delegate_to_c1:
                # model delegation
                c2.delegate(c1, 1, delegations)
                self.hist['total_delegations'] += 1
                hist = c2.eval()
                self.hist['clients'][c2_idx].append((self.last_end_time[c2_idx], hist, list(c1._local_data_dist.keys()))) 
                # self.hist['loss_max'].append(max(self.hist['loss_max'][-1], hist[1]))
                # self.hist['loss_min'].append(min(self.hist['loss_min'][-1], hist[1]))
                
            # self.hist['time_steps'].append(end_t)

            if index != 0 and index % 500 == 0:
                elasped = datetime.datetime.now() - start_time
                rem = elasped / (index+1) * (self.total_number_of_rows-index-1)
                print("\n------------ index {} done ---".format(index), end='') 
                print("elasped time: {}".format(elasped), end='')
                print(" ----  remaining time: {}".format(rem))  

            K.clear_session()
        
        if self._clients[0].is_federated():
            print('start federated simulation')
            self.hist['encountered_clients'] = {}
            for c in self._clients:
                self.hist['encountered_clients'][c._id_num] = c.encountered_clients

            cur_time = 0
            self.hist['clients'] = {}
            for c in self._clients:
                self.hist['clients'][c._id_num] = []
            for _ in tqdm(range(self.hyperparams['num-rounds'])):
                updates = []
                for c in self._clients:
                    c._weights = self.central_server._weights
                    c._weights = c.fit_to(c, 1)
                    hist = c.federated_eval()
                    if c._id_num ==0 :
                        print(hist[1])
                    self.hist['clients'][c._id_num].append((cur_time, hist, 0))
                    updates.append(c._weights)
                agg_weights = list()
                for weights_list_tuple in zip(*updates):
                    agg_weights.append(np.array([np.average(np.array(w), axis=0) for w in zip(*weights_list_tuple)]))
                self.central_server._weights = agg_weights
                cur_time += self.hyperparams['time-per-round']
                del updates
                K.clear_session()

    def register_table(self, *args):
        print('no table used in this class')