import sys
sys.path.insert(0,'..')
import argparse
import datetime
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from clients import get_client_class, LocalClient, GreedySimClient, GreedyNoSimClient, MomentumClient, MomentumWithoutDecayClient
import data_process as dp
import models as custom_models
import copy
import pickle
import numpy as np
from tqdm import tqdm
from get_dataset import get_mnist_dataset, get_cifar_dataset

def main():
    """
    driver for running controlled experiment from one device's perspective
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # parse arguments
    parser = argparse.ArgumentParser(description='set params for controlled experiment')
    parser.add_argument('--out', dest='out_file',
                        type=str, default='logs/log.pickle', help='log history output')
    parser.add_argument('--cfg', dest='config_file',
                        type=str, default='configs/mnist_cfg.json', help='name of the config file')
    parser.add_argument('--draw_graph', dest='graph_file',
                        type=str, default=None, help='name of the output graph filename')               

    parsed = parser.parse_args()

    if parsed.config_file == None or parsed.out_file == None:
        print('Config file and output diretory has to be specified. Run \'python driver.py -h\' for help/.')

    # load config file
    with open(parsed.config_file, 'rb') as f:
        config_json = f.read()
    config = json.loads(config_json)

    if config['dataset'] == 'mnist':
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()
        
    elif config['dataset'] == 'cifar':
        model_fn = custom_models.get_big_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()

    train_data_provider = dp.DataProvider(x_train, y_train_orig)
    test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, 800)

    # get local dataset for clients
    client_label_conf = {}
    for l in config['local-set']:
        client_label_conf[l] = (int) (config['number-of-data-points']/len(config['local-set']))
    x_train_client, y_train_client = train_data_provider.peek(client_label_conf)

    # get pretrained model
    with open(config['pretrained-model'], 'rb') as handle:
        pretrained_model_weight = pickle.load(handle)

    # set params for building clients
    opt_fn = keras.optimizers.SGD
    compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
    hyperparams = config['hyperparams']
    
    # initialize delegation_clients for simulation
    clients = {}
    i = 0
    for k in config['strategies'].keys():
        if config['strategies'][k]:
            client_class = get_client_class(k)
            if client_class == None:
                print("strategy name {} not defined".format(k))
                return

            train_config = {}
            c = client_class(i,
                            model_fn,
                            opt_fn,
                            copy.deepcopy(pretrained_model_weight),
                            x_train_client,
                            y_train_client,
                            test_data_provider,
                            config['goal-set'],
                            compile_config,
                            train_config,
                            hyperparams)
            clients[k] = c
            i += 1
    
    # initialize logs
    logs = {}
    for ck in clients.keys():
        logs[ck] = {}
        logs[ck]['accuracy'] = []
        logs[ck]['loss'] = []

        hist = clients[ck].eval()
        logs[ck]['loss'].append(hist[0])
        logs[ck]['accuracy'].append(hist[1])

    # run simulation
    candidates = np.arange(0,10)
    for i in range(len(config['intervals'])):
        print('simulating range {} of {}'.format(i+1, len(config['intervals'])))
        for _ in tqdm(range(config['intervals'][i])):
            # set labels
            rn = np.random.rand(1)
            if rn > config['noise-percentage']/100.0:  # not noise
                local_labels = config['label-sets'][i]
            else:   # noise
                np.random.shuffle(candidates)
                local_labels = copy.deepcopy(candidates[:len(config['local-set'])])
            label_conf = {}
            for ll in local_labels:
                label_conf[ll] = (int) (config['number-of-data-points']/len(local_labels))

            # print(label_conf)
            x_other, y_other = train_data_provider.peek(label_conf)

            for ck in clients.keys():
                other = get_client_class(ck)(123,   # random id
                                            model_fn,
                                            opt_fn,
                                            copy.deepcopy(pretrained_model_weight),
                                            x_other,
                                            y_other,
                                            test_data_provider,
                                            config['goal-set'],
                                            compile_config,
                                            train_config,
                                            hyperparams)
                clients[ck].delegate(other, 1, 2)
                hist = clients[ck].eval()
                logs[ck]['loss'].append(hist[0])
                logs[ck]['accuracy'].append(hist[1])

    with open(parsed.out_file, 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if parsed.graph_file != None:
        import matplotlib.pyplot as plt
        for k in logs.keys():
            plt.plot(np.arange(0, len(logs[k]['accuracy'])), np.array(logs[k]['accuracy']), lw=1.2)
        plt.legend(list(logs.keys()))
        plt.ylabel("accuracy")
        plt.xlabel("encounters")
        plt.savefig(parsed.graph_file)
        plt.close()
        
if __name__ == '__main__':
    main()