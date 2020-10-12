# -*- coding: utf-8 -*-
import argparse
from pprint import pprint
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import torch
from torch.utils.tensorboard import SummaryWriter

# Add VAE folder to path
import sys
sys.path.append('../vae-lstm/')

# Import network
from network import *
from utils import *
from imshow import *
from read_dataset import *

# Import mcmc methods
from metropolis_hastings_hybrid_kernels import *

# Parser arguments
parser = argparse.ArgumentParser(description='MCMC methods on trained VAE')
parser.add_argument('--sample_size', '--size',
                    type=int, default=100, metavar='N',
                    help='The sample size of "random" numbers.')
parser.add_argument('--burn_in', '--burn',
                    type=int, default=0, metavar='N',
                    help='Number of samples to drop. (default: 0')
parser.add_argument('--log-interval', '--li',
                    type=int, default=50, metavar='N',
                    help='how many batches to wait' +
                         'before logging training status')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--checkpoint', '--check',
                    default='none',
                    help='path to checkpoint to be restored')
parser.add_argument('--predict', '--pred',
                    action='store_true',
                    help='predict test dataset')
parser.add_argument('--plot', '--p',
                    action='store_true',
                    help='plot dataset sample')
parser.add_argument('--summary', '--sm',
                    action='store_true',
                    help='show summary of model')
parser.add_argument('--start',
                    type=float, nargs='+',
                    default=[540.0, 80.0],
                    metavar='x y',
                    help='start of path (default: 540.0 80.0)')
parser.add_argument('--interm',
                    type=float, nargs='+',
                    default=[960.0, 360.0],
                    metavar='x y',
                    help='interm of path (default: 960.0 360.0)')

args = parser.parse_args()


def read_points():
    # Read image
    img = mpimg.imread(args.background_path)
    plt.imshow(img)

    # Read path points from mouse
    pts = plt.ginput(args.num_interm_points + 1)
    plt.close()

    # Save as parameters
    args.start = list(pts[0])
    args.interm = []
    for idx in range(1, len(pts)):
        args.interm.append(list(pts[idx]))

    print('Start : ', args.start)
    print('Interm : ', args.interm)


def plot_trajectory_sample(sample, posterior,
                           name, args,
                           start, interm):

    # Plot sample
    X_smp = [elem[0] for elem in sample]
    Y_smp = [elem[1] for elem in sample]
    plt.plot(X_smp, Y_smp, 'o', alpha=0.4, color='blue',
             label='goal sample')
    plt.legend(loc='upper right')

    # Read image
    img = mpimg.imread(args.background_path)
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

    if args.condition_dimension == 1:
        pass

    if args.condition_dimension == 2:
        # Get max and min of sample
        X_max = args.x_max
        X_min = args.x_min
        Y_max = args.y_max
        Y_min = args.y_min

        # Plot contour
        X_lin = np.linspace(X_min, X_max, 100)
        Y_lin = np.linspace(Y_min, Y_max, 100)

        # Create grid
        X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Evaluate posterior
        Z = np.array([[posterior((cell[0], cell[1]))[0]
                       for cell in row] for row in pos])

        # Plot contour map
        plt.contour(X, Y, Z, 20, cmap='RdGy')

    # Plot image
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


def kde_estimation_goals(trainset):
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True)

    # Iterate through training dataset
    starts = []
    goals = []
    for batch_idx, batch in enumerate(train_loader, 1):
        # Unpack batch
        inputs = batch

        # Send to device
        inputs = inputs.to(args.device)

        # Start and goal points
        start = inputs[:, 0].view(args.batch_size, 2)
        goal = inputs[:, -1].view(args.batch_size, 2)

        # Check condition dimension
        if args.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Append points and add normal noise
        starts.append(start.numpy() +
                      np.random.randn(args.condition_dimension))
        goals.append(goal.numpy() +
                     np.random.randn(args.condition_dimension))

    # Into numpy array
    starts = np.concatenate(starts, axis=0)
    goals = np.concatenate(goals, axis=0)

    # Concatenate
    joint = np.concatenate([goals, starts], axis=1)

    # KDE estimation
    joint_krnl = stats.gaussian_kde(joint.T)
    cond_krnl = stats.gaussian_kde(starts.T)

    # Calculate pdf
    def pdf(x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        z = torch.cat([x, y])
        return joint_krnl.pdf(z) / cond_krnl.pdf(y)

    def logpdf(x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        z = torch.cat([x, y])
        return joint_krnl.logpdf(z) - cond_krnl.logpdf(y)

    # Check condition dimension
    if args.condition_dimension == 1:
        # Plot histograms
        plt.hist(starts, bins='auto', density=True)
        plt.show()

        plt.hist(goals, bins='auto', density=True)
        plt.show()

        # Plot
        x = list(np.linspace(0, 360, 300))
        y = [cond_krnl.pdf(val) for val in x]
        plt.plot(x, y)
        plt.show()

        # Plot contour
        X_lin = np.linspace(0, 360, 100)
        Y_lin = np.linspace(0, 360, 100)

        # Create grid
        X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Evaluate posterior
        Z = np.array([[joint_krnl.pdf((cell[0], cell[1]))[0]
                       for cell in row] for row in pos])

        # Plot contour map
        plt.contour(X, Y, Z, 40, cmap='RdGy')
        plt.show()

        # Construct plot
        fig, ax = plt.subplots()
        ax.set_xlim([0, 360])
        ax.set_ylim([0, 2])

        # Increase margin
        fig.subplots_adjust(bottom=0.2)

        # Slider definition
        def update(val):
            start = slider_start.val

            # Clear axes and set limits again
            ax.clear()
            ax.set_xlim([0, 360])
            ax.set_ylim([0, 2])

            print('\nUpdated slider values !', end='\n')

            # Plot density
            x = list(np.linspace(0, 360, 200))
            y = [pdf(torch.tensor([val]), torch.tensor([start])) for val in x]
            ax.plot(x, y)

        # Plot sliders
        axcolor = 'lightgoldenrodyellow'
        axstart = plt.axes([0.15, 0.09, 0.7, 0.03], facecolor=axcolor)

        slider_start = Slider(axstart, 'start', 0, 360, valinit=0, valstep=1.0)
        slider_start.on_changed(update)
        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

    if args.condition_dimension == 2:
        # Plot estimation

        # Read image
        img = mpimg.imread(args.background_path)
        plt.imshow(img)

        # Get max and min of sample
        X_max = args.x_max
        X_min = args.x_min
        Y_max = args.y_max
        Y_min = args.y_min

        # Plot contour
        X_lin = np.linspace(X_min, X_max, 100)
        Y_lin = np.linspace(Y_min, Y_max, 100)

        # Create grid
        X, Y = np.meshgrid(X_lin, Y_lin, indexing='xy')
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Evaluate posterior
        Z = np.array([[pdf([cell[0], cell[1]], args.start)[0]
                       for cell in row] for row in pos])

        # Plot contour map
        plt.contour(X, Y, Z, 40, cmap='RdGy')
        print('contour_1')
        plt.show()
        print('contour_2')

    return pdf, logpdf


def generate_goal_samples(trainset):
    # Necessary parameters
    max_x = args.x_max
    min_x = args.x_min
    max_y = args.y_max
    min_y = args.y_min

    # Kernel Density estimation of goal dist.
    prior, log_prior = kde_estimation_goals(trainset)

    # Posterior to sample with metropolis hastings
    def posterior(x):
        # Check if x is support
        if args.condition_dimension == 1:
            if x[0] < 0 or x[0] > 360:
                return 0.0

        elif args.condition_dimension == 2:
            if x[0] < min_x or x[0] > max_x or \
               x[1] < min_y or x[1] > max_y:
                return 0.0

        # Reshape interm
        interm = np.array(args.interm, dtype=np.float32).flatten()

        # Fixed values
        goal = torch.tensor([x]).repeat(args.batch_size, 1)
        start = torch.tensor([args.start]).repeat(args.batch_size, 1)
        interm = torch.tensor([interm]).repeat(args.batch_size, 1)

        # Check condition dimension
        if args.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Get random sample from latent space
        latent = torch.randn([args.batch_size, args.latent_dim])
        latent = latent.to(args.device)

        # Generate fake data from generator
        z = torch.cat([latent, start, goal], dim=1)
        decoded_z = args.network.decode(z)

        # Construct path
        if isinstance(args.network, VAE_DELTA):
            # Reshape decoded
            decoded_z = decoded_z.view(args.batch_size,
                                       args.num_interm_points, 2)

            # From delta get coordinates
            decoded_z[:, 0, :] += torch.tensor([args.start]) \
                                       .repeat(args.batch_size, 1)
            for idx in range(1, args.num_interm_points):
                decoded_z[:, idx, :] += decoded_z[:, idx - 1, :]

            # Flatten decoded
            decoded_z = decoded_z.flatten(start_dim=1)

        # Calculate Gaussian
        LOGP = F.mse_loss(decoded_z, interm, reduction='none')
        P = torch.exp(-0.5 * LOGP)

        # Value
        return torch.mean(P).detach().numpy() * \
            prior(goal[0, :], start[0, :])

    def log_posterior(x):
        # Check if x is support
        if args.condition_dimension == 1:
            if x[0] < 0 or x[0] > 360:
                return -10e10

        elif args.condition_dimension == 2:
            if x[0] < min_x or x[0] > max_x or \
               x[1] < min_y or x[1] > max_y:
                return -10e10

        # Reshape interm
        interm = np.array(args.interm, dtype=np.float32).flatten()

        # Fixed values
        goal = (torch.tensor([x], dtype=torch.float32)
                .repeat(args.batch_size, 1))
        start = (torch.tensor([args.start], dtype=torch.float32)
                 .repeat(args.batch_size, 1))
        interm = (torch.tensor([interm], dtype=torch.float32)
                  .repeat(args.batch_size, 1))

        # Check condition dimension
        if args.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Get random sample from latent space
        latent = torch.randn([args.batch_size, args.latent_dim])
        latent = latent.to(args.device)

        # Generate fake data from generator
        z = torch.cat([latent, start, goal], dim=1)
        decoded_z = args.network.decode(z)

        # Construct path
        if isinstance(args.network, VAE_DELTA):
            # Reshape decoded
            decoded_z = decoded_z.view(args.batch_size,
                                       args.num_interm_points, 2)

            # From delta get coordinates
            decoded_z[:, 0, :] += torch.tensor([args.start]) \
                                       .repeat(args.batch_size, 1)
            for idx in range(1, args.num_interm_points):
                decoded_z[:, idx, :] += decoded_z[:, idx - 1, :]

            # Flatten decoded
            decoded_z = decoded_z.flatten(start_dim=1)

        # Calculate logarithm of gaussian
        LOGP = F.mse_loss(decoded_z, interm, reduction='none')

        return torch.mean(LOGP).detach().numpy() + \
            log_prior(goal[0, :], start[0, :])

    # Hybrid kernel proposals

    # Uniform all scene
    def proposal_1(x, x_prime):
        a = stats.uniform.logpdf(x_prime[0], loc=min_x, scale=max_x)
        b = stats.uniform.logpdf(x_prime[1], loc=min_y, scale=max_y)

        return a * b

    def step_1(x):
        x_p = stats.uniform.rvs(loc=min_x, scale=max_x)
        y_p = stats.uniform.rvs(loc=min_y, scale=max_y)

        return (x_p, y_p)

    # Multivariate normal proposal
    sigma = 1

    def proposal_2(x, x_prime):
        SIGMA = [[sigma ** 2, 0],
                 [0, sigma ** 2]]

        return stats.multivariate_normal.logpdf(x_prime, x, SIGMA)

    def step_2(x):
        SIGMA = [[sigma ** 2, 0],
                 [0, sigma ** 2]]

        return stats.multivariate_normal.rvs(x, SIGMA)

    # Normal proposal on X coordinate
    def proposal_3(x, x_prime):
        return stats.norm.logpdf(x_prime[0], x[0], sigma)

    def step_3(x):
        return (stats.norm.rvs(x[0], sigma), x[1])

    # Normal proposal on Y coordinate
    def proposal_4(x, x_prime):
        return stats.norm.logpdf(x_prime[1], x[1], sigma)

    def step_4(x):
        return (x[0], stats.norm.rvs(x[1], sigma))

    # Normal proposal on angle
    def proposal_5(x, x_prime):
        return stats.norm.logpdf(x_prime, x, sigma)

    def step_5(x):
        return [stats.norm.rvs(x, sigma), ]

    # Uniform proposal on angle
    def proposal_6(x, x_prime):
        return stats.uniform.logpdf(x[0], loc=0, scale=360)

    def step_6(x):
        return [stats.uniform.rvs(loc=0, scale=360), ]

    if args.condition_dimension == 1:
        # Intial value for Metropolis-Hastings
        x_init = [np.random.uniform(0, 360), ]

        # Proposals
        proposal = [proposal_5, proposal_6]
        step = [step_5, step_6]
        dist = [.9, .1]

    elif args.condition_dimension == 2:
        # Intial value for Metropolis-Hastings
        x_init = [np.random.uniform(min_x, max_x),
                  np.random.uniform(min_y, max_y)]

        # Proposals
        proposal = [proposal_2, proposal_3, proposal_4]
        step = [step_2, step_3, step_4]
        dist = [.2, .4, .4]

    # Sample using Metropolis-Hastings
    (sample, walk, rejected) = \
        metropolis_hastings_hybrid_kernels(
            args.sample_size,
            x_init,
            log_posterior,
            proposal,
            step,
            dist,
            args.burn_in,
            args.log_interval)

    # Check condition dimension
    if args.condition_dimension == 1:
        angle_sample = [elem[0] for elem in sample]

        plt.title('MCMC Sample')
        plt.hist(angle_sample, bins='auto', density=True)
        plt.show()

        sample = []
        for deg in angle_sample:
            rad = np.deg2rad(deg)
            coord = [100 * np.cos(rad) + 1920.0 / 2.0,
                     -100 * np.sin(rad) + 1080.0 / 2.0]
            bnd = project_to_boundary(
                torch.tensor([coord], dtype=torch.float32))
            sample.append(bnd[0, :].tolist())

    # Plot sample
    name = '../imgs/sample_vae_s={}_b={}_start={}_interm={}.png' \
           .format(args.sample_size, args.burn_in,
                   args.start, len(args.interm))
    plot_trajectory_sample(sample, posterior, name, args,
                           start=args.start, interm=args.interm)
    # name = '../imgs/sample_walk_s={}_b={}_start={}_interm={}.png' \
    #        .format(args.sample_size, args.burn_in,
    #                args.start, len(args.interm))
    # plot_walk(sample, rejected, posterior, name)

    return sample


def main():
    # Printing parameters
    torch.set_printoptions(precision=2)
    torch.set_printoptions(edgeitems=5)

    # Set up GPU
    if args.device != 'cpu':
        args.device = torch.device('cuda:0'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    # Selected device for trainning or inference
    print('device : {}'.format(args.device))

    # Read parameters from checkpoint
    if args.checkpoint:
        read_checkpoint(args)

    # Save parameters in string to name the execution
    args.run = create_run_name(args)

    # print run name
    print('execution name : {}'.format(args.run))

    if not args.predict:
        # Tensorboard summary writer
        writer = SummaryWriter('runs/' + args.run)

        # Save as parameter
        args.writer = writer

    # Read dataset
    trn, vld = load_dataset(args)

    # Get hparams from args
    args.hparams = get_hparams(args.__dict__)
    print('\nParameters :')
    pprint(args.hparams)
    print()

    # Create network
    if args.network == 'vae':
        network = VAE(args)
    elif args.network == 'vae_delta':
        network = VAE_DELTA(args)

    # Send networks to device
    args.network = network.to(args.device)

    # number of parameters
    total_params = sum(p.numel()
                       for p in args.network.parameters()
                       if p.requires_grad)
    print('number of trainable parameters : ', total_params)

    # summarize model layers
    if args.summary:
        print(args.network)
        return

    # Set as inference always
    args.predict = True

    # restore checkpoint
    restore_checkpoint(args)

    # Read path points
    read_points()

    # Predict test
    generate_samples(trn, args)

    # Predict test
    generate_goal_samples(trn)

    # (compatibility issues) Add hparams with metrics to tensorboard
    # args.writer.add_hparams(args.hparams, {'metrics': 0})

    # Delete model + Free memory
    del args.network
    torch.cuda.empty_cache()

    if not args.predict:
        # Close tensorboard writer
        args.writer.close()


if __name__ == "__main__":
    main()
