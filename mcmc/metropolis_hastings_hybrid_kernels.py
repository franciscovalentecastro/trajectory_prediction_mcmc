#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def print_rejection_statistics(count_proposals, count_rejected):
    # Rejections per each proposal
    for idx in range(len(count_proposals)):
        perc = 100 * count_rejected[idx] / count_proposals[idx]
        print('Prop {} : Used {} : Rejected {}: Rej. Percent {:.2f} %'
              .format(idx, count_proposals[idx], count_rejected[idx], perc))

    # All rejections
    perc = 100 * sum(count_rejected) / sum(count_proposals)
    print('All : Used {} : Rejected {}: Rej. Percent {:.2f} %'
          .format(sum(count_proposals), sum(count_rejected), perc))


def plot_individual_hist(sample, name, params=['a', 'b', 'c']):
    # Number of parameters
    n = len(sample[0][1])

    # For each parameter
    for idx in range(n):
        # Hist sample
        smp = np.array([elem[1][idx] for elem in sample])

        plt.hist(smp, bins=20)
        plt.title('Histogram of "{}" sample'.format(params[idx]))
        plt.savefig('{}_{}.png'.format(name, params[idx]),
                    bbox_inches='tight', pad_inches=0)
        plt.show()


def plot_individual_walk_mean(walk, burn_in, name,
                              params=['a', 'b', 'c']):
    # Number of parameters
    n = len(walk[0][1])

    means = []

    # For each parameter
    for idx in range(n):
        # Plot walk
        X_wlk = [elem[0] for elem in walk]
        Y_wlk = np.array([elem[1][idx] for elem in walk])
        Y_mean = [np.mean(Y_wlk[:idx + 1]) for idx in range(len(Y_wlk))]

        plt.plot(X_wlk, Y_mean, '-', alpha=1.0,
                 label='param {}'.format(params[idx]))
        means.append(np.mean(Y_wlk))

        # Format plot
        plt.axvline(x=burn_in, color='r', label='Burn-in')
        plt.legend(loc='upper right')
        plt.savefig('{}_{}.png'.format(name, params[idx]),
                    bbox_inches='tight', pad_inches=0)
        plt.show()

    print('Converged to the following mean ')
    print(means)


def plot_individual_walk(walk, rejected, burn_in,
                         name, params=['a', 'b', 'c']):
    # Number of parameters
    n = len(walk[0][1])

    # For each parameter
    for idx in range(n):
        # Plot walk
        X_wlk = [elem[0] for elem in walk]
        Y_wlk = [elem[1][idx] for elem in walk]

        plt.plot(X_wlk, Y_wlk, '-', alpha=1.0,
                 label='param {}'.format(params[idx]))

        # Plot rejected
        X_rej = [elem[0] for elem in rejected]
        Y_rej = [elem[1][idx] for elem in rejected]
        plt.plot(X_rej, Y_rej, 'x', alpha=.2, color='red',
                 label='rejected')

        # Format plot
        plt.axvline(x=burn_in, color='r', label='Burn-in')
        plt.legend(loc='best')
        plt.savefig('{}_{}.png'.format(name, params[idx]),
                    bbox_inches='tight', pad_inches=0)
        plt.show()


def plot_walk(walk, rejected, posterior, name):
    # Plot walk
    X_wlk = [elem[0] for elem in walk]
    Y_wlk = [elem[1] for elem in walk]
    plt.plot(X_wlk, Y_wlk, '-o', alpha=0.4, color='blue',
             label='accepted')

    # Plot rejected
    X_rej = [elem[1][0] for elem in rejected]
    Y_rej = [elem[1][1] for elem in rejected]
    plt.plot(X_rej, Y_rej, 'x', alpha=0.2, color='red',
             label='rejected')
    plt.legend(loc='upper right')

    # Get max and min of walk
    X_max = 1920  # np.max(X_wlk + X_rej)
    X_min = 0     # np.min(X_wlk + X_rej)
    Y_max = 1080  # np.max(Y_wlk + Y_rej)
    Y_min = 0     # np.min(Y_wlk + Y_rej)

    # Plot contour
    X_lin = np.linspace(X_min, X_max, 100)
    Y_lin = np.linspace(Y_min, Y_max, 100)

    # Create grid
    X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Evaluate posterior
    Z = np.array([[posterior((cell[0], cell[1]))
                   for cell in row] for row in pos])

    # Read image
    img = mpimg.imread('../mini-vae/imgs/train_station.jpg')
    plt.imshow(img)

    # Plot contour map
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_sample(sample, posterior, name, args,
                params=['a', 'b'], start=None, interm=None):

    # Plot sample
    X_smp = [elem[0] for elem in sample]
    Y_smp = [elem[1] for elem in sample]
    plt.plot(X_smp, Y_smp, 'o', alpha=0.4, color='blue',
             label='sample')
    plt.legend(loc='upper right')

    # Get max and min of sample
    X_max = 1920  # np.max(X_smp)
    X_min = 0     # np.min(X_smp)
    Y_max = 1080  # np.max(Y_smp)
    Y_min = 0     # np.min(Y_smp)

    # Plot contour
    X_lin = np.linspace(X_min, X_max, 100)
    Y_lin = np.linspace(Y_min, Y_max, 100)

    # Create grid
    X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Evaluate posterior
    Z = np.array([[posterior((cell[0], cell[1]))
                   for cell in row] for row in pos])

    # Read image
    img = mpimg.imread('../mini-vae/imgs/train_station.jpg')
    plt.imshow(img)

    # Plot start
    plt.plot(start[0], start[1], marker='x',
             color='red', mew=2, ms=10)
    plt.annotate('start', xy=start, xytext=(10, -5),
                 textcoords='offset points', color='white')

    # Plot intermediate
    for idx, pt in enumerate(interm):
        plt.plot(pt[0], pt[1], marker='x',
                 color='red', mew=2, ms=10)
        plt.annotate('interm_{}'.format(idx), xy=pt, xytext=(10, -5),
                     textcoords='offset points', color='white')

    # Plot contour map
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.xlabel(params[0])
    plt.ylabel(params[1])
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


# Acceptance Ratio
def log_acceptance_ratio(x_p, x_t, log_posterior, log_proposal):
    return min(0, log_posterior(x_p) + log_proposal(x_p, x_t) -
               log_posterior(x_t) - log_proposal(x_t, x_p))


def metropolis_hastings_hybrid_kernels(sample_size,
                                       x_initial,
                                       log_posterior,
                                       log_proposal,
                                       step,
                                       kernel_probs,
                                       burn_in,
                                       log_interval):
    # Current step
    t = 0

    # Number of proposal
    len_proposal = len(log_proposal)

    # Initial uniform sample
    x_t = x_initial

    # Sample
    sample = []

    # Random walk
    walk = []
    rejected = []

    # Counting statistics
    count_proposals = [0] * len_proposal
    count_rejected = [0] * len_proposal

    # Init Burn-in count
    burnt = 0

    # Sample until desired number
    while len(sample) < sample_size:
        # Save random walk
        walk.append(x_t)

        # Select transition kernel
        kernel_idx = np.random.choice(len_proposal, p=kernel_probs)

        log_proposal_t = log_proposal[kernel_idx]
        step_t = step[kernel_idx]

        # Generate random step from proposed distribution
        x_p = step_t(x_t)

        # Calculate the acceptance ratio
        rho = np.exp(log_acceptance_ratio(x_p, x_t,
                                          log_posterior,
                                          log_proposal_t))

        # Random uniform sample
        u = np.random.uniform()

        if u < rho:  # Accept
            x_t = x_p

        else:  # Reject
            rejected.append((t, x_p))
            if burnt == burn_in:
                count_rejected[kernel_idx] += 1

        if burnt == burn_in:  # Sample stage
            sample.append(x_t)
            count_proposals[kernel_idx] += 1

            if len(sample) % log_interval == 0:
                print('# Samples:', len(sample))

        else:  # Burn-in stage
            burnt += 1

            if burnt % log_interval == 0:
                print('# Burnt:', burnt)

        # Next step
        t += 1

    # Rejection statistics
    print_rejection_statistics(count_proposals, count_rejected)

    return (sample, walk, rejected)


def sample_from_normal_posterior(rho):
    # Parser arguments
    parser = argparse.ArgumentParser(
        description='Metropolis_Hastings generator.')
    parser.add_argument('--sample_size', '--size',
                        type=int, default=100, metavar='N',
                        help='The sample size of "random" numbers.')
    parser.add_argument('--burn_in', '--burn',
                        type=int, default=0, metavar='N',
                        help='Number of samples to drop. (default: 0')
    parser.add_argument('--sigma', '--sd',
                        type=float, default=.1, metavar='N',
                        help='Standard deviation of normal step (default: .1)')
    parser.add_argument('--log-interval', '--li',
                        type=int, default=100, metavar='N',
                        help='interval to print current status')
    args = parser.parse_args()

    # Posterior distribution params
    mu_1 = -.5
    mu_2 = .5
    sigma_1 = sigma_2 = 1

    # Construct vectors
    MU = [mu_1, mu_2]
    SIGMA = [[sigma_1 ** 2, rho * sigma_1 * sigma_2],
             [rho * sigma_1 * sigma_2, sigma_2 ** 2]]

    # Posterior to sample with metropolis hastings
    def posterior(x):
        return stats.multivariate_normal.pdf(x, MU, SIGMA)

    def log_posterior(x):
        return stats.multivariate_normal.logpdf(x, MU, SIGMA)

    # Hybrid kernel proposals

    # X_1 | X_2 normal
    def proposal_1(x, x_prime):
        mean = mu_1 + rho * (sigma_1 / sigma_2) * (x[1] - mu_2)
        sigma = (sigma_1 ** 2) * (1 - rho ** 2)

        return stats.norm.logpdf(x_prime[1], mean, sigma)

    def step_1(x):
        mean = mu_1 + rho * (sigma_1 / sigma_2) * (x[1] - mu_2)
        sigma = (sigma_1 ** 2) * (1 - rho ** 2)

        return (stats.norm.rvs(mean, sigma), x[1])

    # X_2 | X_1 normal
    def proposal_2(x, x_prime):
        mean = mu_2 + rho * (sigma_2 / sigma_1) * (x[0] - mu_1)
        sigma = (sigma_2 ** 2) * (1 - rho ** 2)

        return stats.norm.logpdf(x_prime[0], mean, sigma)

    def step_2(x):
        mean = mu_2 + rho * (sigma_2 / sigma_1) * (x[0] - mu_1)
        sigma = (sigma_2 ** 2) * (1 - rho ** 2)

        return (x[0], stats.norm.rvs(mean, sigma))

    # Intial value for Metropolis-Hastings
    x_init = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(args.sample_size,
                                           x_init,
                                           log_posterior,
                                           [proposal_1, proposal_2],
                                           [step_1, step_2],
                                           [.5, .5],
                                           args.burn_in,
                                           args.log_interval)

    # Plot sample
    name = 'imgs/sample_normal_s={}_b={}_r={}.png'\
           .format(args.sample_size, args.burn_in, rho)
    plot_sample(sample, posterior, name)
    name = 'imgs/walk_normal_s={}_b={}_r={}.png'\
           .format(args.sample_size, args.burn_in, rho)
    plot_walk(sample, rejected, posterior, name)
    name = 'imgs/burn-in_normal_s={}_b={}_r={}.png'\
           .format(args.sample_size, args.burn_in, rho)
    plot_individual_walk_mean(list(enumerate(walk)), name)
    plot_individual_hist(list(enumerate(sample)), '')

    print('Kolmogorov-Smirnov test of normality :')
    s0 = [elem[0] for elem in sample]
    s1 = [elem[1] for elem in sample]

    print(stats.kstest(s0, 'norm'))
    print(stats.kstest(s1, 'norm'))

    return sample


def main():
    # Print format to 3 decimal spaces and fix seed
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Sample from posterior with different parameters
    sample_from_normal_posterior(rho=0.8)
    sample_from_normal_posterior(rho=0.99)


if __name__ == "__main__":
    main()
