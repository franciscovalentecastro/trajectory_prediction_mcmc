from IPython.display import display, HTML
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# Initial parameters
obsmat_txt = '../vae-lstm/datasets/ewap_dataset/seq_eth/obsmat.txt'
H_txt = '../vae-lstm/datasets/ewap_dataset/seq_eth/H.txt'

# Load train. Number of intermediate points plus start and goal
# dataset = ETH(obsmat_txt, H_txt, 5 + 2)

# Trajectory list
trajectories = []

# Read homography matrix
# H = np.loadtxt(H_txt)

# Read text file obsmat.txt
names = ['frame_number', 'pedestrian_ID',
         'pos_x', 'pos_z', 'pos_y',
         'v_x', 'v_z', 'v_y']
obsmat = pd.read_csv(obsmat_txt, delimiter=r"\s+",
                     names=names, header=None,
                     engine='python')

diff_ped_id = set(list(obsmat['pedestrian_ID']))

for ped_id in diff_ped_id:
    print(idx)

    # Get only one path
    ped_path = obsmat.query('pedestrian_ID == {}'.format(ped_id))
    
    # Get x, y values
    x = ped_path['pos_x']
    y = ped_path['pos_y']

    # Number of points
    n = len(x)

    # Too small for cubic interpolation
    if n <= 3:
        continue

    # Interpolate and parametrize
    t = np.linspace(0, 1, n)
    x_curve = interp1d(t, x, kind='cubic')
    y_curve = interp1d(t, y, kind='cubic')

    # Create path
    t_path = np.linspace(0, 1, path_length)
    x_path = x_curve(t_path)
    y_path = y_curve(t_path)

    # Pair of list
    points = list(zip(x_path, y_path))

    if belong_to_path(x_path, y_path):
        # New trajectory
        trajectories.append(points)

# Convert into numpy array
trajectories = np.array(trajectories, dtype=np.float32)
