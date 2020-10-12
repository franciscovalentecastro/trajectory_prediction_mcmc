#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def point_inside_rectangle(px, py, rx_1, ry_1, rx_2, ry_2):
    if px > rx_2 or px < rx_1 or py > ry_2 or py < ry_1:
        return False

    return True


def point_inside_area(x, y, area):
    # Get areas
    areas = pd.read_csv(
        '../mini-vae/datasets/CentralStation_areasDescriptions.csv')
    areas = areas.to_numpy()

    # Get rectangle
    rx_1 = areas[area, 2]
    ry_1 = areas[area, 3]
    rx_2 = areas[area, 4]
    ry_2 = areas[area, 7]

    # Check for elements
    resp_1 = point_inside_rectangle(x, y,
                                    rx_1,
                                    ry_1,
                                    rx_2,
                                    ry_2)

    return resp_1


def belong_to_path(x, y):
    # Parameters of path
    start = [0]
    goal = [6, 4, 8, 9, 3, 2, 7, 1]

    # Flag start and goal
    flg_start = False
    flg_goal = False

    # Check starting areas
    for s in start:
        flg_start = flg_start or point_inside_area(x[0], y[0], s)

    # Check starting areas
    for g in goal:
        flg_goal = flg_goal or point_inside_area(x[-1], y[-1], g)

    return flg_start and flg_goal


def main():
    # Parser arguments
    parser = argparse.ArgumentParser(description='Read Dataset')
    parser.add_argument('--number-of-points', '--n',
                        type=int, default=100, metavar='N',
                        help='number of trajectories to generate.')
    args = parser.parse_args()

    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Trajectory list
    trajectories = []

    # Read text file
    with open('datasets/CentralStation_trainingSet.txt') as file:
        lines = file.readlines()

        for line in tqdm(lines, 'read file'):
            values = np.array([int(elem) for elem in line.split()])
            values = values.reshape([-1, 3])

            # Recover x, y, t variables from file
            x = values[:, 0].tolist()
            y = values[:, 1].tolist()
            t = values[:, 2].tolist()

            # New trajectory
            trajectories.append([x, y, t])

    # Read image
    img = mpimg.imread('imgs/train_station.jpg')
    plt.imshow(img)

    # Plot data
    count = 0
    for idx, sample in tqdm(enumerate(trajectories), 'filter trajectories'):
        # Get x and y
        x = sample[0]
        y = sample[1]

        # Only fixed number
        if belong_to_path(x, y):
            # Plot single trajectory
            plt.plot(x, y, color='blue', alpha=.3)

            # New trajectory in path
            count += 1

        # Print traj stats
        # print('Trajectory #{}'.format(idx))
        # print('Length : {}'.format(len(x)))

    print('# of Trajectories {}'.format(count))

    # Plot areas boxes
    areas = pd.read_csv('datasets/CentralStation_areasDescriptions.csv')

    for idx in range(len(areas)):
        x1 = areas['1'].iloc[idx]
        y1 = areas['2'].iloc[idx]
        x2 = areas['3'].iloc[idx]
        y2 = areas['4'].iloc[idx]
        x3 = areas['5'].iloc[idx]
        y3 = areas['6'].iloc[idx]
        x4 = areas['7'].iloc[idx]
        y4 = areas['8'].iloc[idx]

        plt.plot([x1, x2, x4, x3, x1], [y1, y2, y4, y3, y1], color='red')
        plt.annotate('idx = {}'.format(idx), xy=(x1, y1),
                     xycoords='data', color='white')

    plt.show()


if __name__ == "__main__":
    main()
