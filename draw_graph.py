import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def get_accs_over_time(loaded_hist, key):
    loss_diff_at_time = []
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

def get_accs_and_minmax_over_time(loaded_hist, key, window_size):
    # get first accuracies
    accum = []
    for c in loaded_hist[key].keys():
        accum.append(loaded_hist[key][c][0][1][1])
    start_mean = sum(accum)/len(accum)
    
    loss_diff_at_time = []
    for k in loaded_hist[key].keys():
        i = 0
        for t, h, _ in loaded_hist[key][k]:
            if t != 0:
                loss_diff_at_time.append((t, loaded_hist[key][k][i][1][1] - loaded_hist[key][k][i-1][1][1]))
            i += 1
    
    lst = len(loss_diff_at_time)
    loss_diff_at_time.sort(key=lambda x: x[0])
    ldat_nodup = []
    for lt in loss_diff_at_time:
        if len(ldat_nodup) != 0 and ldat_nodup[-1][0] == lt[0]:
            ldat_nodup[-1] = (ldat_nodup[-1][0], ldat_nodup[-1][1] + lt[1])
        else:
            ldat_nodup.append(lt)
    times = []
    loss_list = []
    times.append(0)
    
    loss_list.append(start_mean)
    for i in range(1, len(ldat_nodup)):
        times.append(ldat_nodup[i][0])
        loss_list.append(loss_list[i-1] + ldat_nodup[i][1]/len(loaded_hist[key]))
        
    # for clients
    clients_accs_diffs = {} # stores accumulated difference
    for k in loaded_hist[key].keys():
        clients_accs_diffs[k] = [(0, start_mean)]

        i = 0
        for t, h, _ in loaded_hist[key][k]:
            if t != 0:
                clients_accs_diffs[k].append(
                    (t, clients_accs_diffs[k][-1][1] + 
                     (loaded_hist[key][k][i][1][1] - loaded_hist[key][k][i-1][1][1])
                     ))
            i += 1
            
    # populate empty rows
    cad_filled = {}
    for k in clients_accs_diffs.keys():
        cad_filled[k] = [(0, start_mean)]
        j = 0
        for i in range(len(times)):
            while j < len(clients_accs_diffs[k]) and clients_accs_diffs[k][j][0] < times[i]:
                j += 1
            cad_filled[k].append((times[i], clients_accs_diffs[k][j-1][1]))
            
    # get mins and maxs
    mm_times = []
    mins = []
    maxs = []
    for t in times:
        cur_min = 100
        cur_max = -100
        for c in cad_filled.keys(): # for clients
            for e in cad_filled[c]:
                if e[0] >= t - window_size and e[0] < t + window_size:
                    cur_min = min(cur_min, e[1])
                    cur_max = max(cur_max, e[1])
                # @TODO what if nothing got searched in the window?
        if cur_min != 100 and cur_max != -100:
            mm_times.append(t)
            mins.append(cur_min)
            maxs.append(cur_max)
    
    # for i in range(len(mm_times)):
    #     maxs[i] = (maxs[i] - loss_list[i])/len(loaded_hist[key])
    #     mins[i] = (mins[i] - loss_list[i])/len(loaded_hist[key])
    
    mm_times.append(times[-1])
    mins.append(mins[-1])
    maxs.append(maxs[-1])

    for i in range(10):
        print('{}: {}:{}'.format(mm_times[i], mins[i], maxs[i]))
        
    return times, loss_list, mm_times, mins, maxs

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='set params for controlled experiment')
    parser.add_argument('--hist', dest='log_file',
                        type=str, default=None, help='log file')
    parser.add_argument('--out', dest='graph_file',
                        type=str, default='figs/figure.pdf', help='output figure name')
    parser.add_argument('--metrics', dest='metrics',
                        type=str, default='loss-and-accuracy', help='metrics')
    parser.add_argument('--minmax', dest='minmax', action='store_true', default=False, help='minmax')
    parsed = parser.parse_args()  

    with open(parsed.log_file, 'rb') as handle:
        hists = pickle.load(handle)

    # if parsed.metrics == 'loss-and-accuracy':
    #     key = 'accuracy'
    # elif parsed.metrics == 'f1-score-weighted':
    #     key = 'f1-score'
    # else:
    #     ValueError('invalid metrics: {}'.format(parsed.metrics))
    
    print('drawing graph...')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    if parsed.minmax:
        processed_hists = {}
        for k in hists.keys():
            print('processing {}'.format(k))
            t, acc, mt, mins, maxs = get_accs_and_minmax_over_time(hists[k], 'clients', 2)
            processed_hists[k] = {}
            processed_hists[k]['times'] = t
            processed_hists[k]['accs'] = acc
            processed_hists[k]['mm_times'] = mt
            processed_hists[k]['mins'] = mins
            processed_hists[k]['maxs'] = maxs

        fig, ax = plt.subplots()
        for k in processed_hists.keys():
            ax.plot(np.array(processed_hists[k]['times']), np.array(processed_hists[k]['accs']), lw=1.2)
            ax.fill_between(processed_hists[k]['mm_times'], np.array(processed_hists[k]['mins'])
                                , np.array(processed_hists[k]['maxs']), alpha=0.2)
        plt.legend(list(processed_hists.keys()))
        plt.ylabel("acc")
        plt.xlabel("time")
        plt.savefig(parsed.graph_file)
        plt.close()
        return

    processed_hists = {}
    for k in hists.keys():
        t, acc = get_accs_over_time(hists[k], 'clients')
        processed_hists[k] = {}
        processed_hists[k]['times'] = t
        processed_hists[k]['accs'] = acc

    for k in processed_hists.keys():
        plt.plot(np.array(processed_hists[k]['times']), np.array(processed_hists[k]['accs']), lw=1.2)
    plt.legend(list(processed_hists.keys()))
    plt.ylabel("Accuracy")
    plt.xlabel("Time")
    plt.savefig(parsed.graph_file)
    plt.close()

if __name__ == '__main__':
    main()

