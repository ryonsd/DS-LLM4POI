import numpy as np
import pandas as pd
import random
import json
import argparse
from fastdtw import fastdtw
from tqdm import tqdm
import os


def calc_dtw(test_traj, train_traj):
    test_traj = test_traj[["Latitude", "Longitude"]].values
    train_traj = train_traj[["Latitude", "Longitude"]].values

    distance, _ = fastdtw(test_traj, train_traj)
    return distance

def calc_lcs(test_traj, train_traj):
    train_traj = train_traj["PoiId"].tolist()
    test_traj = test_traj["PoiId"].tolist()

    # Calculate the LCSS distance
    n = len(test_traj)
    m = len(train_traj)
    lcss = np.zeros((n+1, m+1))

    for i in range(1, n+1):
        for j in range(1, m+1):
            if test_traj[i-1] == train_traj[j-1]:
                lcss[i][j] = lcss[i-1][j-1] + 1
            else:
                lcss[i][j] = max(lcss[i-1][j], lcss[i][j-1])

    return lcss[n][m]

# def calc_edit_dist(test_traj, train_traj):
#     test_traj = test_traj["PoiId"].tolist()
#     train_traj = train_traj["PoiId"].tolist()

#     # Calculate the edit distance
#     n = len(test_traj)
#     m = len(train_traj)
#     dp = np.zeros((n+1, m+1))

#     for i in range(1, n+1):
#         dp[i][0] = i
#     for j in range(1, m+1):
#         dp[0][j] = j

#     for i in range(1, n+1):
#         for j in range(1, m+1):
#             if test_traj[i-1] == train_traj[j-1]:
#                 dp[i][j] = dp[i-1][j-1]
#             else:
#                 dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

#     return dp[n][m]

def calc_jaccard(traj1, traj2):
    traj1_poi_list = traj1["PoiId"].tolist()
    traj2_poi_list = traj2["PoiId"].tolist()
    number_of_common_poi = len(
        set(traj1_poi_list).intersection(set(traj2_poi_list)))
    number_of_total_poi = len(set(traj1_poi_list).union(set(traj2_poi_list)))
    return number_of_common_poi / number_of_total_poi

#------------------------------------------------------------
def run(test_data, traj_ids, train_data, index, data_dir, method, suffix):
    distance_test = {}
    for traj_id in tqdm(traj_ids[:]):
        test_data_sample = test_data[test_data['pseudo_session_trajectory_id'] == traj_id]
        test_traj = test_data_sample.iloc[:-1]

        if len(test_traj) != 0:
            distance = {}
            for trajectory_id in train_data['pseudo_session_trajectory_id'].unique():
                train_traj = train_data[train_data['pseudo_session_trajectory_id'] == trajectory_id].iloc[:-1].copy()

                if method == 'dtw':
                    dist = calc_dtw(test_traj, train_traj)
                elif method == 'lcs':
                    dist = calc_lcs(test_traj, train_traj)
                # elif method == 'edit':
                #     dist = calc_edit_dist(test_traj, train_traj)
                elif method == 'jaccard':
                    dist = calc_jaccard(test_traj, train_traj)
                else:
                    raise ValueError('Invalid method')

                distance[str(trajectory_id)] = dist
        else:
            distance = {}
            train_traj_ids = train_data['pseudo_session_trajectory_id'].unique()
            train_traj_ids = random.sample(list(train_traj_ids), len(train_traj_ids))
            for trajectory_id in train_traj_ids:
                distance[str(trajectory_id)] = 0

        distance_test[str(traj_id)] = distance

    # Save the result as json
    output_dir = f'{data_dir}/similarity/{method}/{suffix}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f'{output_dir}/{method}_{index}.json', 'w') as f:
        json.dump(distance_test, f)

#====================================================================================================
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str)
    argparser.add_argument('-m', type=str, default="dtw")
    argparser.add_argument('-i', type=int, default=0)

    args = argparser.parse_args()
    data_dir = args.d
    method = args.m
    index = args.i

    # Load the preprocessed data
    data_dir = f"datasets//{data_dir}/preprocessed"
    train_data = pd.read_csv(data_dir + '/train_sample.csv')
    # validate_data = pd.read_csv(data_dir + '/validate_sample.csv')
    test_data = pd.read_csv(data_dir + '/test_sample_with_traj.csv')

    start_index = 2*index
    end_index = 2*(index+1)
    print(start_index, end_index)

    # Calculate the distance between the trajectories
    test_traj_ids = test_data['pseudo_session_trajectory_id'].unique()[start_index:end_index]
    if len(test_traj_ids) > 0:
        run(test_data, test_traj_ids, train_data, index, data_dir, method, suffix='test')