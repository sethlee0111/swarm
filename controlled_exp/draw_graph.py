import pickle
import argparse
import numpy as np

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='set params for controlled experiment')
    parser.add_argument('--hist', dest='log_file',
                        type=str, default=None, help='log file')
    parser.add_argument('--out', dest='graph_file',
                        type=str, default='figs/figure.pdf', help='output figure name')
    parser.add_argument('--metrics', dest='metrics',
                        type=str, default='loss-and-accuracy', help='metrics')

    parsed = parser.parse_args()  

    with open(parsed.log_file, 'rb') as handle:
        logs = pickle.load(handle)

    if parsed.metrics == 'loss-and-accuracy':
        key = 'accuracy'
    elif parsed.metrics == 'f1-score-weighted':
        key = 'f1-score'
    else:
        ValueError('invalid metrics: {}'.format(parsed.metrics))
    
    if parsed.graph_file != None:
        import matplotlib.pyplot as plt
        for k in logs.keys():
            plt.plot(np.arange(0, len(logs[k][key])), np.array(logs[k][key]), lw=1.2)
        plt.legend(list(logs.keys()))
        # plt.ylim(0.9, 0.935)
        plt.ylabel(key)
        plt.xlabel("encounters")
        plt.savefig(parsed.graph_file)
        plt.close()

if __name__ == '__main__':
    main()