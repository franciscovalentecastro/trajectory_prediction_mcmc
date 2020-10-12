import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# Utilities library
from read_dataset import *


class Trajectories(Dataset):
    def __init__(self, path_csv):
        # Print message
        print('\nStarted loading Trajectories.', end='\n')

        # Load csv
        self.trajectories = pd.read_csv(path_csv).to_numpy()

        # Print message
        print('\nTrajectories was successfully loaded.', end='\n')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        element = [[self.trajectories[idx, 1],
                    self.trajectories[idx, 2]],
                   [self.trajectories[idx, 3],
                    self.trajectories[idx, 4]],
                   [self.trajectories[idx, 5],
                    self.trajectories[idx, 6]]]

        return torch.tensor(element)


class GrandCentralStation(Dataset):
    def __init__(self, path_txt, path_length):
        # Print message
        print('\nStarted loading Grand Central Station dataset.', end='\n')

        # If already calculated
        if os.path.exists('tmp/trajectories_{}.npy'
                          .format(path_length)):
            # Load npy array
            self.trajectories = np.load('tmp/trajectories_{}.npy'
                                        .format(path_length))

        else:
            # Trajectory list
            self.trajectories = []

            # Read text file
            with open(path_txt) as file:
                lines = file.readlines()

                for line in tqdm(lines, 'dataset'):
                    values = np.array([int(elem) for elem in line.split()])
                    values = values.reshape([-1, 3])

                    # Recover x, y, t variables from file
                    x = values[:, 0]
                    y = values[:, 1]

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
                        self.trajectories.append(points)

            # Convert into numpy array
            self.trajectories = np.array(self.trajectories, dtype=np.float32)

            # Save npy array
            np.save('tmp/trajectories_{}.npy'
                    .format(path_length), self.trajectories)

        # Print message
        print('\nGrand Central Station was successfully loaded.', end='\n')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.tensor(self.trajectories[idx])


class ETH(Dataset):
    def __init__(self, obsmat_txt, H_txt, path_length):
        # Print message
        print('\nStarted loading ETH dataset.', end='\n')

        # If already calculated
        if os.path.exists('../vae-lstm/tmp/eth_trajectories_{}.npy'
                          .format(path_length)):
            # Load npy array
            self.trajectories = np.load('../vae-lstm/tmp/eth_trajectories_{}.npy'
                                        .format(path_length))

        else:
            # Trajectory list
            self.trajectories = []

            # Read homography matrix
            H = np.loadtxt(H_txt)
            Hinv = np.linalg.inv(H)

            # Read text file obsmat.txt
            names = ['frame_number', 'pedestrian_ID',
                     'pos_x', 'pos_z', 'pos_y',
                     'v_x', 'v_z', 'v_y']
            obsmat = pd.read_csv(obsmat_txt, delimiter=r"\s+",
                                 names=names, header=None,
                                 engine='python')

            diff_ped_id = set(list(obsmat['pedestrian_ID']))

            for ped_id in diff_ped_id:
                # Get only one path
                ped_path = obsmat.query('pedestrian_ID == {}'.format(ped_id))

                # Get x, y values
                wrld_x = np.array(ped_path['pos_x'])
                wrld_y = np.array(ped_path['pos_y'])

                # Transform points to image space
                cat = np.vstack([wrld_x, wrld_y, np.ones_like(wrld_x)]).T
                tCat = (Hinv @ cat.T).T

                # Get points in image
                x = tCat[:, 1] / tCat[:, 2]
                y = tCat[:, 0] / tCat[:, 2]

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

                # Apend to trajectory list
                self.trajectories.append(points)

            # Convert into numpy array
            self.trajectories = np.array(self.trajectories, dtype=np.float32)

            # Save npy array
            np.save('../vae-lstm/tmp/eth_trajectories_{}.npy'
                    .format(path_length), self.trajectories)

        # Print message
        print('\nETH was successfully loaded.', end='\n')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.tensor(self.trajectories[idx])


class HOTEL(Dataset):
    def __init__(self, obsmat_txt, H_txt, path_length):
        # Print message
        print('\nStarted loading HOTEL dataset.', end='\n')

        # If already calculated
        if os.path.exists('../vae-lstm/tmp/hotel_trajectories_{}.npy'
                          .format(path_length)):
            # Load npy array
            self.trajectories = np.load('../vae-lstm/tmp/hotel_trajectories_{}.npy'
                                        .format(path_length))

        else:
            # Trajectory list
            self.trajectories = []

            # Read homography matrix
            H = np.loadtxt(H_txt)
            Hinv = np.linalg.inv(H)

            # Read text file obsmat.txt
            names = ['frame_number', 'pedestrian_ID',
                     'pos_x', 'pos_z', 'pos_y',
                     'v_x', 'v_z', 'v_y']
            obsmat = pd.read_csv(obsmat_txt, delimiter=r"\s+",
                                 names=names, header=None,
                                 engine='python')

            diff_ped_id = set(list(obsmat['pedestrian_ID']))

            for ped_id in diff_ped_id:
                # Get only one path
                ped_path = obsmat.query('pedestrian_ID == {}'.format(ped_id))

                # Get x, y values
                wrld_x = np.array(ped_path['pos_x'])
                wrld_y = np.array(ped_path['pos_y'])

                # Transform points to image space
                cat = np.vstack([wrld_x, wrld_y, np.ones_like(wrld_x)]).T
                tCat = (Hinv @ cat.T).T

                # Get points in image
                x = tCat[:, 1] / tCat[:, 2]
                y = tCat[:, 0] / tCat[:, 2]

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

                # Apend to trajectory list
                self.trajectories.append(points)

            # Convert into numpy array
            self.trajectories = np.array(self.trajectories, dtype=np.float32)

            # Save npy array
            np.save('../vae-lstm/tmp/hotel_trajectories_{}.npy'
                    .format(path_length), self.trajectories)

        # Print message
        print('\nHOTEL was successfully loaded.', end='\n')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.tensor(self.trajectories[idx])
