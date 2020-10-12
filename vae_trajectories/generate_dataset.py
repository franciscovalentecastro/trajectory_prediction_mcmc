#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Parser arguments
parser = argparse.ArgumentParser(description='Generate Dataset')
parser.add_argument('--number-of-points', '--n',
                    type=int, default=100, metavar='N',
                    help='number of trajectories to generate.')
parser.add_argument('--sigma',
                    type=float, default=.01, metavar='sd',
                    help='sd of noise added (default: .01)')
parser.add_argument('--optimize',
                    action='store_true',
                    help='opimtize estimates with least-squares')
args = parser.parse_args()


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Names of columns
    names = ['P1_x', 'P1_y',
             'P2_x', 'P2_y',
             'P3_x', 'P3_y']

    # Distribution parameters
    mean = [10.0, -10.0]
    sigma = 1.0

    # Generate data points
    data = []
    for idx in range(args.number_of_points):

        # Pick a random mean
        mu = random.choice(mean)

        data.append([-10, 0,
                     0, np.random.normal(mu, sigma),
                     10, 0])

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=names)

    # Write dataframe to csv
    df.to_csv('trajectories.csv')

    # Plot data
    for sample in data:
        # Get x and y
        x = [sample[0], sample[2], sample[4]]
        y = [sample[1], sample[3], sample[5]]

        # Plot single trajectory
        plt.plot(x, y)

    plt.show()


if __name__ == "__main__":
    main()
