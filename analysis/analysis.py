#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parser arguments
parser = argparse.ArgumentParser(
    description='Gradient descent algorithm.')
parser.add_argument('--log-interval', '--li',
                    type=int, default=1000, metavar='N',
                    help='interval to print current status')
pargs = parser.parse_args()


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # List to store information
    x = []
    y = []
    c = []

    # Read samples file
    file = open('out.txt')
    for line in file:

        splt = line.split()

        c.append(float(splt[1]))
        x.append(float(splt[2]))
        y.append(float(splt[3]))

    # Close file
    file.close()

    col = ['r', 'g', 'b', 'y', 'g', 'k']

    for idx in range(len(c) // 4):
        # Plot samples
        index = 4 * idx
        plt.plot(x[index: index + 4],
                 y[index: index + 4],
                 '*-', color=col[idx % 6])

    plt.show()


if __name__ == "__main__":
    main()
