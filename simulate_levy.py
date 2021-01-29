import numpy as np 
from scipy.stats import uniform
from scipy.stats import levy
import pandas as pd
import argparse

N = 2000
MAX_XY = 60
QUAD_SIZE = (int) ((2 * MAX_XY) / 3)
THRES_LOC = 70
THRES_TIME = 250
ALPHA = 0.8
BETA = 1.5
THRES_NEIGHBOR = 3
TIME_STEP_SIZE = 0.5

# data frame column names
C1_POS_X = "x1"
C1_POS_Y = "y1"
C2_POS_X = "x2"
C2_POS_Y = "y2"
DIST = "distance"
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

def main():
    parser = argparse.ArgumentParser(description='set params for simulation')
    parser.add_argument('--clients', dest='clients',
                        type=int, default=45, help='number of clients')
    parser.add_argument('--episodes', dest='episodes',
                        type=int, default=10, help='number of episodes')
    parser.add_argument('--duration', dest='duration',
                        type=int, default=100, help='duration of one episode')
    parser.add_argument('--filename', dest='filename', type=str, default=None, help='output filename')
    parser.add_argument('--disp_graph', dest='graph',
                        type=bool, default=False, help='display the graph for simulated nodes')
    parsed = parser.parse_args()

    if parsed.filename == None:
        print('Output filename has to be specified. Run \'python simulate_levy.py -h\' for help/.')

    df, _ = create_dataset(parsed.clients, parsed.episodes, parsed.duration)

    df.to_pickle(parsed.filename)

    print("encounter file generated: {}".format(parsed.filename))


def truncated_levy(n, thres, alpha):
    r = np.zeros(1)
    while r.size < n:
        r = levy.rvs( size=(int)(2 * n) )
        r = np.power(r, -(1+alpha))
        r = r[r < thres]
        r = r[:n]
    return r

def time_to_tread_path(p1, p2, k, rho):
    x = np.linalg.norm(p1 - p2)
    return k * np.power(x, (1-rho))

def levy_walk_not_interpolated(n, max_xy, x_orig, y_orig, thres_loc, thres_time, alpha, beta):
    K = 1.
    RHO = .5
    # uniformly distributed angles
    angle = uniform.rvs( size=(n,), loc=.0, scale=2.*np.pi )

    # levy distributed step length
    rv = truncated_levy(n, thres_loc, alpha)
    stay_time = truncated_levy(n, thres_time, beta)
    
    # x and y coordinates
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x_orig
    y[0] = y_orig
    for i in range(1,n):
        x[i] = x[i-1] + rv[i] * np.cos(angle[i])
        if x[i] > max_xy:
            x[i] -= 2 * (x[i] - max_xy)
        elif x[i] < -max_xy:
            x[i] -= 2 * (x[i] + max_xy)
            
        y[i] = y[i-1] + rv[i] * np.sin(angle[i])
        if y[i] > max_xy:
            y[i] -= 2 * (y[i] - max_xy)
        elif y[i] < -max_xy:
            y[i] -= 2 * (y[i] + max_xy)
    points = np.array(list(zip(x,y)))
    
    # calculate elasped times
    times = np.zeros(2 * points.shape[0] - 1)
    times[0] = stay_time[0]
    for i in range(1, points.shape[0]):
        times[2*i-1] = times[2*(i-1)] + time_to_tread_path(points[i-1], points[i], K, RHO)
        times[2*i] = times[2*i-1] + stay_time[i]

    return points, times

def levy_walk(n, max_xy, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta):
    K = 1.
    RHO = .5
    # uniformly distributed angles
    angle = uniform.rvs( size=(n,), loc=.0, scale=2.*np.pi )

    # levy distributed step length
    rv = truncated_levy(n, thres_loc, alpha)
    stay_time = truncated_levy(n, thres_time, beta)
    
    # x and y coordinates
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x_orig
    y[0] = y_orig
    # indices when it hit the wall. later used to interpolate points
    x_wall = set()
    y_wall = set()
    for i in range(1,n):
        x[i] = x[i-1] + rv[i] * np.cos(angle[i])
        if x[i] > max_xy:
            x_wall.add(i) 
            x[i] -= 2 * (x[i] - max_xy)

        elif x[i] < -max_xy:
            x_wall.add(i) 
            x[i] -= 2 * (x[i] + max_xy)
            
        y[i] = y[i-1] + rv[i] * np.sin(angle[i])
        if y[i] > max_xy:
            y_wall.add(i)
            y[i] -= 2 * (y[i] - max_xy)
        elif y[i] < -max_xy:
            y_wall.add(i)
            y[i] -= 2 * (y[i] + max_xy)
            
    points = np.array(list(zip(x,y)))
 
    # start constructing all the interpolations
    agg_points = np.empty([0,2])
    agg_times = np.empty([0,1])
    cur_t = 0
    for i in range(points.shape[0]-1):
        if i in x_wall or i in y_wall:
            if i in x_wall:
                hit_point = np.array([points[i][0], (points[i][1]+points[i][1])/2])
            elif i in y_wall:
                hit_point = np.array([(points[i][0]+points[i][0])/2, points[i][1]])
            # from i to hit point
            travel_time = time_to_tread_path(points[i], hit_point, K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(points[i], hit_point, interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)
            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size
            # from hit point to i
            travel_time = time_to_tread_path(hit_point, points[i+1], K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(hit_point, points[i+1], interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)
            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size
        else:
            travel_time = time_to_tread_path(points[i], points[i+1], K, RHO)
            interp_pieces = (int)(travel_time/time_step_size)
            interp_vals_p = np.linspace(points[i], points[i+1], interp_pieces)
            interp_vals_t = np.arange(cur_t, cur_t+travel_time, time_step_size)

            agg_points = np.concatenate((agg_points, interp_vals_p), axis=0)
            agg_times = np.append(agg_times, interp_vals_t)
            cur_t += interp_pieces * time_step_size

    return agg_points, agg_times

def levy_walk_episodes(n, max_xy, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta, episodes):
    points, times = levy_walk(n, max_xy, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta)
    for _ in range(episodes-1):
        p, t = levy_walk(n, max_xy, x_orig, y_orig, thres_loc, thres_time, time_step_size, alpha, beta)
        points = np.concatenate((points, p), axis=0)
        t = np.insert(t, 0, times[-1])
        t += times[-1]
        times = np.concatenate((times, t), axis=0)
    return points,times
    
def create_dataset(num_clients=100,
                    time_steps=20,
                    episodes=5,
                    file_name=None):
    pt_tuples = []
    start_point_x = []
    start_point_y = []
    for x in range(-MAX_XY, MAX_XY, QUAD_SIZE):
        for y in range(-MAX_XY, MAX_XY, QUAD_SIZE):
            for _ in range((int)(num_clients/9)):
                start_point_x.append((np.random.rand(1)*QUAD_SIZE+x)[0])
                start_point_y.append((np.random.rand(1)*QUAD_SIZE+y)[0])

    for i in range(len(start_point_x)):
        print("client {}: {}, {}".format(i, start_point_x[i], start_point_y[i]))
        thres_loc = THRES_LOC
        alpha = ALPHA
        frac = 4./5. # fraction of relativly static nodes
        if i % (num_clients / 9) <= (num_clients / 9) * frac:
            thres_loc = thres_loc / 5
            alpha = alpha * 2
        pt_tuples.append(levy_walk_episodes(time_steps, 
                    MAX_XY,
                    start_point_x[i],
                    start_point_y[i],
                    thres_loc, 
                    THRES_TIME, 
                    TIME_STEP_SIZE,
                    alpha, 
                    BETA,
                    episodes))
    

    dtypes = np.dtype([
          (ENC_IDX, int),
          (CLIENT1, int),
          (CLIENT2, int),
          (TIME_START, float),
          (TIME_END, float)
          ])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data = data)

    for c1 in range(num_clients):
        print("----- processing encounters from client {}".format(c1))
        for c2 in range(c1+1, num_clients):
            # print("----- processing encounters btwn client {} and {}".format(c1, c2))
            add_encounters(df, pt_tuples, c1, c2)

    df = df.groupby([ENC_IDX,CLIENT1,CLIENT2])\
       .agg({TIME_START:'min', TIME_END:'max'})\
       .sort_values(by=[TIME_START,CLIENT1,TIME_END,ENC_IDX,CLIENT2])\
       .reset_index()

    df = df.astype({ENC_IDX: int, CLIENT1: int, CLIENT2: int})
    
    return df, pt_tuples

def add_encounters(df, pt_tuples, c1, c2):
    """
    pt_tuples[client_num][1][i] is a time when the client has started moving
    from pt_tuples[client_num][0][i] to [i+1]
    we assume c1 is less than c2
    """
    def _dist(c1, c2, te1, te2):
        return np.linalg.norm(_get_pos(c1, te1) - _get_pos(c2, te2))

    def _is_encounter():
        if _get_start_time(c2, te2) < _get_time(c1, te1) and _get_start_time(c1, te1) < _get_time(c2, te2):
            return _dist(c1, c2, te1, te2) < THRES_NEIGHBOR
        return False

    def _get_time(client, idx):
        return pt_tuples[client][1][idx]

    def _get_start_time(client, idx):
        if idx == 0:
            return 0
        return pt_tuples[client][1][idx-1]

    def _get_pos(client, idx):
        return pt_tuples[client][0][idx]

    def _put_encounter(time_start, time_end):
        df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = \
                        {CLIENT1: c1,
                         CLIENT2: c2,
                         ENC_IDX: enc_idx,
                         TIME_START: time_start,
                         TIME_END: time_end}

    # index for time_end
    te1 = 0
    te2 = 0
    prev_enc_idx_c1 = 0
    prev_enc_idx_c2 = 0

    enc_idx = 0

    c1_len = len(pt_tuples[c1][0]) 
    c2_len = len(pt_tuples[c2][0]) 

    while te1 < c1_len and te2 < c2_len:
        # @TODO: currently this loop doesn't record the last encounter
        if _is_encounter():
            enc_start_time = max(_get_start_time(c1, te1), _get_start_time(c2, te2))
            enc_end_time = min(_get_time(c1, te1), _get_time(c2, te2))
            if prev_enc_idx_c1 != te1 - 1 and prev_enc_idx_c2 != te2 - 1:
                enc_idx += 1
            _put_encounter(enc_start_time, enc_end_time)
            prev_enc_idx_c1 = te1
            prev_enc_idx_c2 = te2

        if te1 == c1_len - 1:
            te2 += 1
        elif te2 == c2_len - 1:
            te1 += 1
        else:
            # increment one of two pointers
            if _get_time(c1, te1) < _get_time(c2, te2):
                te1 += 1
            else:
                te2 += 1
    
    return df

if __name__ == '__main__':
    main()