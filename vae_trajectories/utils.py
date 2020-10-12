# -*- coding: utf-8 -*-
import torch
import warnings
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.image as mpimg

# Import network
from network import *
from datasets import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def get_max(x, dim=(2, 3)):
    b = x.shape[0]
    j = x.shape[1]
    d = x.shape[2]
    m = x.view(b, j, -1).argmax(1)
    indices = torch.cat(((m // d).view(-1, 1),
                         (m % d).view(-1, 1)),
                        dim=1)
    return indices


def plot_grandcentral(inpt):
    # Read image
    img = mpimg.imread('imgs/train_station.jpg')
    plt.imshow(img)

    # Plot all trajectories
    for idx in range(inpt.shape[0]):
        # Get x and y values
        x = inpt[idx, :, 0].tolist()
        y = inpt[idx, :, 1].tolist()

        # Plot trajectory
        plt.plot(x, y, color='black', alpha=.4)

    # Start and goal points
    start = inpt[:, 0].flatten(start_dim=1)
    goal = inpt[:, -1].flatten(start_dim=1)

    # Project to project_to_boundary
    pStart = project_to_boundary(start)
    pGoal = project_to_boundary(goal)

    plt.scatter(pStart[:, 0].tolist(),
                pStart[:, 1].tolist(),
                marker='x', color='white')

    plt.scatter(pGoal[:, 0].tolist(),
                pGoal[:, 1].tolist(),
                marker='x', color='green')

    # Plot edge of scene
    x = [450, 1420, 1900, 35, 450]
    y = [170, 170, 1000, 1000, 170]
    plt.plot(x, y, 'r.-', label='border')

    # Plot center of scene
    c_x = 1920.0 / 2.0
    c_y = 1080.0 / 2.0
    plt.scatter(c_x, c_y, marker='x', label='center')
    plt.legend()

    plt.show()


def plot_eth(inpt):
    # Read image
    img = mpimg.imread('datasets/ewap_dataset/seq_eth/snap.png')
    plt.imshow(img)

    # Plot all trajectories
    for idx in range(inpt.shape[0]):
        # Get x and y values
        x = inpt[idx, :, 0].tolist()
        y = inpt[idx, :, 1].tolist()

        # Plot trajectory
        plt.plot(x, y, color='green', alpha=1)

        # Plot direction arrows
        for jdx in range(1, len(x)):
            plt.arrow(x[jdx - 1], y[jdx - 1],
                      x[jdx] - x[jdx - 1],
                      y[jdx] - y[jdx - 1],
                      color='green',
                      shape='full',
                      lw=1,
                      length_includes_head=True,
                      head_width=5)
    plt.show()


def plot_hotel(inpt):
    # Read image
    img = mpimg.imread('datasets/ewap_dataset/seq_hotel/snap.png')
    plt.imshow(img)

    # Plot all trajectories
    for idx in range(inpt.shape[0]):
        # Get x and y values
        x = inpt[idx, :, 0].tolist()
        y = inpt[idx, :, 1].tolist()

        # Plot trajectory
        plt.plot(x, y, color='green', alpha=1)

        # Plot direction arrows
        for jdx in range(1, len(x)):
            plt.arrow(x[jdx - 1], y[jdx - 1],
                      x[jdx] - x[jdx - 1],
                      y[jdx] - y[jdx - 1],
                      color='green',
                      shape='full',
                      lw=1,
                      length_includes_head=True,
                      head_width=5)
    plt.show()


def load_dataset(args):

    if args.dataset == 'traj':
        # Initial parameters
        path_csv = '../vae-lstm/datasets/trajectories.csv'

        # Load train
        dataset = Trajectories(path_csv)

        # Coordinates
        args.x_max = 1
        args.x_min = 0
        args.y_max = 1
        args.y_min = 0

    elif args.dataset == 'gc':
        # Initial parameters
        path_txt = '../vae-lstm/datasets/CentralStation_trainingSet.txt'

        # Load train. Number of intermediate points plus start and goal
        dataset = GrandCentralStation(path_txt, args.num_interm_points + 2)

        # Coordinates
        args.x_max = 1920
        args.x_min = 0
        args.y_max = 1080
        args.y_min = 0

        # Image files
        args.background_path = '../vae-lstm/imgs/train_station.jpg'

    elif args.dataset == 'eth':
        # Initial parameters
        obsmat_txt = '../vae-lstm/datasets/ewap_dataset/seq_eth/obsmat.txt'
        H_txt = '../vae-lstm/datasets/ewap_dataset/seq_eth/H.txt'

        # Load train. Number of intermediate points plus start and goal
        dataset = ETH(obsmat_txt, H_txt, args.num_interm_points + 2)

        # Coordinates
        args.x_max = 640
        args.x_min = 0
        args.y_max = 480
        args.y_min = 0

        # Image files
        args.background_path = '../vae-lstm/datasets/ewap_dataset/' + \
                               'seq_eth/snap.png'

    elif args.dataset == 'hotel':
        # Initial parameters
        obsmat_txt = '../vae-lstm/datasets/ewap_dataset/seq_hotel/obsmat.txt'
        H_txt = '../vae-lstm/datasets/ewap_dataset/seq_hotel/H.txt'

        # Load train. Number of intermediate points plus start and goal
        dataset = HOTEL(obsmat_txt, H_txt, args.num_interm_points + 2)

        # Coordinates
        args.x_max = 720
        args.x_min = 0
        args.y_max = 576
        args.y_min = 0

        # Image files
        args.background_path = '../vae-lstm/datasets/ewap_dataset/' + \
                               'seq_hotel/snap.png'

    train_size = int(args.train_percentage * len(dataset))
    valid_size = len(dataset) - train_size
    trn, vld = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Dataset information
    print('train dataset : {} elements'.format(len(trn)))
    print('validate dataset : {} elements'.format(len(vld)))

    return trn, vld


def get_hparams(dictionary):
    hparams = {}
    for key, value in dictionary.items():
        if isinstance(value, int) or \
           isinstance(value, str) or \
           isinstance(value, float) or \
           isinstance(value, list):
            hparams[key] = value
    return hparams


def read_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Read weights from {}.'.format(args.checkpoint))

    # Discard hparams
    discard = ['run', 'predict', 'checkpoint', 'summary']

    # Restore past checkpoint
    hparams = checkpoint['hparams']
    for key, value in hparams.items():
        if (key not in discard):
            args.__dict__[key] = value


def restore_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore weights
    args.network.load_state_dict(checkpoint['state_dict'])

    if args.predict:
        # To do inference
        args.network.eval()
    else:
        # Read optimizer parameters
        args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # To continue training
        args.network.train()


def process_checkpoint(loss, global_step, args):
    # check if current batch had best generating fitness
    steps_before_best = 100
    if loss < args.best and global_step > steps_before_best:
        args.best = loss

        # Save best checkpoint
        torch.save({
            'state_dict': args.network.state_dict(),
            'optimizer_state_dict': args.optimizer.state_dict(),
            'hparams': args.hparams,
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/loss', loss, global_step)

    # Save current checkpoint
    torch.save({
        'state_dict': args.network.state_dict(),
        'optimizer_state_dict': args.optimizer.state_dict(),
        'hparams': args.hparams,
    }, "checkpoint/last_{}.pt".format(args.run))


def create_run_name(args):
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}={}'.format('pts', args.num_interm_points)
    run += '_{}={}'.format('cond', args.condition_dimension)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run


def add_tensorboard(inputs, targets, outputs, global_step, name='Train'):
    # Make targets and output slices
    trgt_slice = targets.sum(dim=1, keepdim=True)
    otpt_slice = outputs.sum(dim=1, keepdim=True)

    trgt_htmp = heatmap(trgt_slice).to(args.device)
    otpt_htmp = heatmap(otpt_slice).to(args.device)

    # Make grids
    image_grid = make_grid(inputs, nrow=4, padding=2, pad_value=1)
    trgt_slice_grid = make_grid(trgt_slice, nrow=4, padding=2, pad_value=1)
    otpt_slice_grid = make_grid(otpt_slice, nrow=4, padding=2, pad_value=1)
    trgt_htmp_grid = make_grid(trgt_htmp, nrow=4, padding=2, pad_value=1)
    otpt_htmp_grid = make_grid(otpt_htmp, nrow=4, padding=2, pad_value=1)

    # Create Heatmaps grid
    args.writer.add_image('{}/gt'.format(name), trgt_htmp_grid, global_step)
    args.writer.add_image('{}/gt_image'.format(name),
                          image_grid + trgt_slice_grid, global_step)
    args.writer.add_image('{}/pred'.format(name), otpt_htmp_grid, global_step)
    args.writer.add_image('{}/pred_image'.format(name),
                          image_grid + otpt_slice_grid, global_step)


def generate_samples(trainset, args):
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True)

    # Generate data points
    data = []
    for batch_idx, batch in enumerate(train_loader, 1):
        # Unpack batch
        inputs = batch

        # Send to device
        inputs = inputs.to(args.device)

        # Get random sample from latent space
        latent = torch.randn((args.batch_size, args.latent_dim))
        latent = latent.to(args.device)

        # Start and goal points
        start = inputs[:, 0].view(args.batch_size, 2)
        goal = inputs[:, -1].view(args.batch_size, 2)

        # Check condition dimension
        if args.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Generate fake data from generator
        z = torch.cat([latent, start, goal], dim=1)
        decoded_z = args.network.decode(z)

        # Check condition dimension
        if args.condition_dimension == 1:
            start = project_to_boundary(inputs[:, 0].view(args.batch_size, 2))
            goal = project_to_boundary(inputs[:, -1].view(args.batch_size, 2))

        # Construct result
        result = torch.cat([start, decoded_z, goal], dim=1)
        result = result.view(args.batch_size,
                             args.num_interm_points + 2, 2)
        result = result.detach().cpu().numpy()

        # Construct path
        if isinstance(args.network, VAE_DELTA):
            # From delta get coordinates
            for idx in range(1, args.num_interm_points + 1):
                result[:, idx, :] += result[:, idx - 1, :]

        # Append to data list
        for jdx in range(args.batch_size):
            data.append([result[jdx, :, 0], result[jdx, :, 1]])

    if args.dataset == 'gc':
        # Read image
        plt_color = 'black'
        img = mpimg.imread('../vae-lstm/imgs/train_station.jpg')
        plt.imshow(img)

    elif args.dataset == 'eth':
        # Read image
        plt_color = 'white'
        img = mpimg.imread('../vae-lstm/datasets/ewap_dataset/'
                           'seq_eth/snap.png')
        plt.imshow(img)

    elif args.dataset == 'hotel':
        # Read image
        plt_color = 'white'
        img = mpimg.imread('../vae-lstm/datasets/ewap_dataset/'
                           'seq_hotel/snap.png')
        plt.imshow(img)

    # Plot data
    for sample in data:
        # Get x and y
        x = sample[0]
        y = sample[1]

        # Plot single trajectory
        plt.plot(x, y, color=plt_color, alpha=.4)

    if hasattr(args, 'start'):
        # Plot start
        plt.plot(args.start[0], args.start[1], marker='x',
                 color='red', mew=2, ms=10)
        plt.annotate('start', xy=args.start, xytext=(10, -5),
                     textcoords='offset points', color='white')

    if hasattr(args, 'interm'):
        # Plot intermediate
        for idx, pt in enumerate(args.interm):
            plt.plot(pt[0], pt[1], marker='x',
                     color='red', mew=2, ms=10)
            plt.annotate('interm_{}'.format(idx), xy=pt, xytext=(10, -5),
                         textcoords='offset points', color='white')

    plt.show()
