import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from timeit import default_timer as timer
import matplotlib
import copy
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import models as custom_models
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset
from clients import get_client_class
import pickle
import argparse
from swarm import Swarm
import data_process as dp
from swarm_utils import get_time
import data_process as dp
import boto3
from cfg_utils import setup_env, LOG_FOLDER, FIG_FOLDER, HIST_FOLDER
from pathlib import PurePath, Path
import logging

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

def main():
    setup_env()

    # parse arguments
    parser = argparse.ArgumentParser(description='set params for simulation')
    parser.add_argument('--seed', dest='seed',
                        type=int, default=0, help='use pretrained weights')

    parser.add_argument('--tag', dest='tag',
                        type=str, default='default_tag', help='tag')
    parser.add_argument('--cfg', dest='config_file',
                        type=str, default='toy_realworld_mnist_cfg.json', help='name of the config file')

    parsed = parser.parse_args()

    if parsed.config_file == None or parsed.tag == None:
        print('Config file and the tag has to be specified. Run \'python delegation_swarm_driver.py -h\' for help/.')
        
    LOG_FILE_PATH = Path(LOG_FOLDER, parsed.tag + '.log')
    if LOG_FILE_PATH.exists():
        ans = input("Simulation under the same tag already exists. Do you want to proceed? [y/N]: ")
        if not (ans == 'y' or ans == 'Y'):
            print('exit simulation.')
            exit()
    try:
        with open('configs/workstation_info.json', 'rb') as f:
            wsinfo_json = f.read()
        wsinfo = json.loads(wsinfo_json)
        wsinfo['workstation-name']
    except:
        print("create file \'configs/workstation_info.json\'")

    logging.basicConfig(filename=LOG_FILE_PATH, filemode='w', 
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    np.random.seed(parsed.seed)
    tf.compat.v1.set_random_seed(parsed.seed)

    # load config file
    with open(parsed.config_file, 'rb') as f:
        config_json = f.read()
    config = json.loads(config_json)

    logging.info('-----------------------<config file>-----------------------')
    for k in config:  
        logging.info(str(k + ':'))
        logging.info('    ' + str(config[k]))

    if config['dataset'] == 'mnist':
        num_classes = 10
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()
        
    elif config['dataset'] == 'cifar':
        num_classes = 10
        model_fn = custom_models.get_big_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()
    elif config['dataset'] == 'opportunity-uci':
        model_fn = custom_models.get_deep_conv_lstm_model
        x_train, y_train_orig, x_test, y_test_orig = get_opp_uci_dataset('data/opportunity-uci/oppChallenge_gestures.data',
                                                                         SLIDING_WINDOW_LENGTH,
                                                                         SLIDING_WINDOW_STEP)
    else:
        print("invalid dataset name")
        return

    CLIENT_NUM = config['client-num']

    # use existing pretrained model
    if config['pretrained-model'] != None:
        print("using existing pretrained model")
        with open(config['pretrained-model'], 'rb') as handle:
            init_weights = pickle.load(handle)
    # pretrain new model
    else:
        # pretraining setup
        x_pretrain, y_pretrain_orig = dp.filter_data_by_labels(x_train, y_train_orig, 
                                                    np.arange(num_classes),
                                                                config['pretrain-config']['data-size'])
        y_pretrain = keras.utils.to_categorical(y_pretrain_orig, num_classes)

        pretrain_config = {'batch_size': 50, 'shuffle': True}
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        init_model = model_fn()
        init_model.compile(**compile_config)
        pretrain_config['epochs'] = config['pretrain-setup']['epochs']
        pretrain_config['x'] = x_pretrain
        pretrain_config['y'] = y_pretrain
        pretrain_config['verbose'] = 1
        init_model.fit(**pretrain_config)
        init_weights = init_model.get_weights()
        with open('remote_hist/pretrained_model_2nn_local_updates_'+get_time({})+'_.pickle', 'wb') as handle:
            pickle.dump(init_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    enc_config = config['enc-exp-config']
    enc_exp_config = {}
    enc_exp_config['data_file_name'] = enc_config['encounter-data-file']
    enc_exp_config['communication_time'] = enc_config['communication-time']
    enc_exp_config['train_time_per_step'] = enc_config['train-time-per-step']
    try:
        enc_exp_config['max_rounds'] = enc_config['max-rounds']
    except:
        raise ValueError('no \'max-rounds\' found in the config file (replaces max-delegations)')
    # if config['mobility-model'] == 'levy-walk':
    try:
        enc_exp_config['local_data_per_quad'] = config['district-9']
    except:
        enc_exp_config['local_data_per_quad'] = None

    hyperparams = config['hyperparams']

    test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['hyperparams']['test-data-per-label'])

    test_swarms = []
    swarm_names = []

    # OPTIMIZER = keras.optimizers.SGD

    orig_swarm = Swarm(model_fn,
                       keras.optimizers.SGD,
                       LocalClient,
                       CLIENT_NUM,
                       init_weights,
                       x_train,
                       y_train_orig,
                       test_data_provider,
                       config['local-set-size'],
                       config['goal-set-size'],
                       config['local-data-size'],
                       enc_exp_config,
                       hyperparams
                       )

    for k in config['strategies'].keys():
        if config['strategies'][k]:
            swarm_names.append(k)
            client_class = get_client_class(k)
            test_swarms.append(
                Swarm(
                    model_fn,
                    keras.optimizers.SGD,
                    client_class,
                    CLIENT_NUM,
                    init_weights,
                    x_train,
                    y_train_orig,
                    test_data_provider,
                    config['local-set-size'],
                    config['goal-set-size'],
                    config['local-data-size'],
                    enc_exp_config,
                    hyperparams,
                    orig_swarm
                )
            )
    
    # del orig_swarm

    hists = {}
    for i in range(0, len(test_swarms)):
        start = timer()
        print("{} == running {} with {}".format(swarm_names[i], test_swarms[i].__class__.__name__, test_swarms[i]._clients[0].__class__.__name__))
        print("swarm {} of {}".format(i+1, len(test_swarms)))
        test_swarms[i].run()
        end = timer()
        print('-------------- Elasped Time --------------')
        print(end - start)
        hists[swarm_names[i]] = (test_swarms[i].hist)

        hist_file_path = PurePath(HIST_FOLDER, 'partial_{}_'.format(i) + parsed.tag + '.pickle')
        if i > 0:
            os.remove(PurePath(HIST_FOLDER, 'partial_{}_'.format(i-1) + parsed.tag + '.pickle'))
        if i == len(test_swarms) - 1:
            hist_file_path = PurePath(HIST_FOLDER, parsed.tag + '.pickle')
        with open(hist_file_path, 'wb') as handle:
            pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('drawing graph...')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    processed_hists = {}
    for k in hists.keys():
        # if 'federated' in k:
        #     continue
        t, acc = get_accs_over_time(hists[k], 'clients')
        processed_hists[k] = {}
        processed_hists[k]['times'] = t
        processed_hists[k]['accs'] = acc

    for k in processed_hists.keys():
        # if 'federated' in k:
        #     continue
        plt.plot(np.array(processed_hists[k]['times']), np.array(processed_hists[k]['accs']), lw=1.2)
    plt.legend(list(processed_hists.keys()))
    if hyperparams['evaluation-metrics'] == 'f1-score-weighted':
        plt.ylabel("F1-score")
    else:
        plt.ylabel("Accuracy")
    plt.xlabel("Time")
    graph_file_path = PurePath(FIG_FOLDER, parsed.tag + '.pdf')
    plt.savefig(graph_file_path)
    plt.close()

    logging.info('Simulation completed successfully.')

    # upload to S3 storage
    client = boto3.client('s3')
    S3_BUCKET_NAME = 'opfl-sim-models'
    upload_log_path = PurePath(wsinfo['workstation-name'], 'logs/' + parsed.tag + '.log')
    client.upload_file(str(LOG_FILE_PATH), S3_BUCKET_NAME, str(upload_log_path))
    upload_hist_path = PurePath(wsinfo['workstation-name'], 'hists/' + parsed.tag + '.pickle')
    client.upload_file(str(hist_file_path), S3_BUCKET_NAME, str(upload_hist_path))
    upload_graph_path = PurePath(wsinfo['workstation-name'], 'figs/' + parsed.tag + '.pdf')
    client.upload_file(str(graph_file_path), S3_BUCKET_NAME, str(upload_graph_path))

def get_accs_over_time(loaded_hist, key):
    loss_diff_at_time = []
#     print("total exchanges: {}".format(loaded_hist['total_exchanges'][-1]))
    for k in loaded_hist[key].keys():
        i = 0
        for t, h, _ in loaded_hist[key][k]:
            if t != 0:
                loss_diff_at_time.append((t, loaded_hist[key][k][i][1][1] - loaded_hist[key][k][i-1][1][1]))
            i += 1
    loss_diff_at_time.sort(key=lambda x: x[0])

    # concatenate duplicate time stamps
    ldat_nodup = []
    for lt in loss_diff_at_time:
        if len(ldat_nodup) != 0 and ldat_nodup[-1][0] == lt[0]:
            ldat_nodup[-1] = (ldat_nodup[-1][0], ldat_nodup[-1][1] + lt[1])
        else:
            ldat_nodup.append(lt)
    times = []
    loss_list = []
    times.append(0)
    # get first accuracies
    accum = []
    for c in loaded_hist[key].keys():
        accum.append(loaded_hist[key][c][0][1][1])
        
    loss_list.append(sum(accum)/len(accum))
    for i in range(1, len(ldat_nodup)):
        times.append(ldat_nodup[i][0])
        loss_list.append(loss_list[i-1] + ldat_nodup[i][1]/len(loaded_hist[key]))
        
    return times, loss_list

if __name__ == '__main__':
    main()
