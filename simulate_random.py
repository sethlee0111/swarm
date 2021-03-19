import numpy as np 
from scipy.stats import uniform
from scipy.stats import levy
import pandas as pd
import argparse

# data frame column names
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

def main():
    parser = argparse.ArgumentParser(description='set params for simulation')
    parser.add_argument('--clients', dest='clients',
                        type=int, default=45, help='number of clients')
    parser.add_argument('--encounters', dest='encounters',
                        type=int, default=10, help='number of encounters')
    parser.add_argument('--filename', dest='filename', type=str, default=None, help='output filename')
    parser.add_argument('--disp_graph', dest='graph',
                        type=bool, default=False, help='display the graph for simulated nodes')
    parsed = parser.parse_args()

    if parsed.filename == None:
        print('Output filename has to be specified. Run \'python simulate_levy.py -h\' for help/.')

    df = create_dataset(parsed.clients, parsed.encounters, parsed.filename)

    df.to_pickle(parsed.filename)

    print("encounter file generated: {}".format(parsed.filename))

def create_dataset(num_clients=45,
                    encounters=20,
                    file_name=None):
    dtypes = np.dtype([
          (ENC_IDX, int),
          (CLIENT1, int),
          (CLIENT2, int),
          (TIME_START, float),
          (TIME_END, float)
          ])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data = data)

    cur_time = 0
    for e in range(encounters):
        client1 = np.random.randint(num_clients)
        client2 = np.random.randint(num_clients)
        while client1 == client2:
            client2 = np.random.randint(num_clients)
        if client1 < client2:
            add_encounters(df, cur_time, cur_time+10, e, client1, client2)
        else:
            add_encounters(df, cur_time, cur_time+10, e, client2, client1)   
        cur_time += 10

    df = df.groupby([ENC_IDX,CLIENT1,CLIENT2])\
       .agg({TIME_START:'min', TIME_END:'max'})\
       .sort_values(by=[TIME_START,CLIENT1,TIME_END,ENC_IDX,CLIENT2])\
       .reset_index()

    df = df.astype({ENC_IDX: int, CLIENT1: int, CLIENT2: int})
    
    return df

def add_encounters(df, enc_start_time, enc_end_time, enc_idx, c1, c2):
    def _put_encounter(time_start, time_end):
        df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = \
                        {CLIENT1: c1,
                         CLIENT2: c2,
                         ENC_IDX: enc_idx,
                         TIME_START: time_start,
                         TIME_END: time_end}

    _put_encounter(enc_start_time, enc_end_time)

    return df

if __name__ == '__main__':
    main()