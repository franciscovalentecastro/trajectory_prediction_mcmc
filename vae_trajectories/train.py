# -*- coding: utf-8 -*-
import argparse
from pprint import pprint

import torch
import torch.optim as optim
import torch.autograd as autograd

from torch.utils.tensorboard import SummaryWriter

# Import network
from network import *
from utils import *
from imshow import *
from read_dataset import *

# Parser arguments
parser = argparse.ArgumentParser(description='Train trajectory prediction'
                                             'distribution with VAE-LSTM')
parser.add_argument('--train-percentage', '--t',
                    type=float, default=.9, metavar='N',
                    help='porcentage of the training set to use (default: .9)')
parser.add_argument('--batch-size', '--b',
                    type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--log-interval', '--li',
                    type=int, default=50, metavar='N',
                    help='how many batches to wait' +
                         'before logging training status')
parser.add_argument('--epochs', '--e',
                    type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--device', '--d',
                    default='cpu', choices=['cpu', 'cuda'],
                    help='pick device to run the training (defalut: "cpu")')
parser.add_argument('--network', '--net',
                    default='vae',
                    choices=['vae', 'vae_delta'],
                    help='pick a specific network to train'
                         '(default: "vae")')
parser.add_argument('--latent-dim', '--ld',
                    type=int, default=10, metavar='N',
                    help='dimension of latent space (default: 10)')
parser.add_argument('--optimizer', '--o',
                    default='adam', choices=['adam', 'sgd', 'lbfgs'],
                    help='pick a specific optimizer (default: "adam")')
parser.add_argument('--learning-rate', '--lr',
                    type=float, default=1e-3, metavar='N',
                    help='learning rate of optimizer (default: 1E-3)')
parser.add_argument('--hidden-size', '--h-size',
                    type=int, default=1024, metavar='N',
                    help='size of network intermediate layer (default: 1024)')
parser.add_argument('--dataset', '--data',
                    default='traj',
                    choices=['traj', 'gc', 'eth', 'hotel'],
                    help='pick a specific dataset (default: "traj")')
parser.add_argument('--num-interm-points', '--num-pts',
                    type=int, default=1, metavar='N',
                    help='number of intermediate points (default: 1)')
parser.add_argument('--condition-dimension', '--cond',
                    type=int, default=2, metavar='N',
                    help='dimension of conditional variables (default: 2)')
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
args = parser.parse_args()


def batch_status(batch_idx, inputs, outputs,
                 epoch, train_loader, loss, validset):
    # Global step
    global_step = batch_idx + len(train_loader) * epoch

    # update running loss statistics
    args.train_loss += loss
    args.running_loss += loss

    # Write tensorboard statistics
    args.writer.add_scalar('Train/loss', loss, global_step)

    # print every args.log_interval of batches
    if global_step % args.log_interval == 0:
        # validate
        # vloss = validate(validset, log_info=True, global_step=global_step)

        # Process current checkpoint
        process_checkpoint(loss, global_step, args)

        print('Epoch : {} Batch : {} [{}/{} ({:.0f}%)]\n'
              '====> Loss : {:.6f}'
              .format(epoch, batch_idx,
                      args.batch_size * batch_idx,
                      args.dataset_size,
                      100. * batch_idx / args.dataloader_size,
                      args.running_loss / args.log_interval),
              end='\n\n')

        args.running_loss = 0.0

    # (compatibility issues) Pass all pending items to disk
    # args.writer.flush()


def train(trainset, validset):
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True)
    args.dataset_size = len(train_loader.dataset)
    args.dataloader_size = len(train_loader)

    # get some random training images
    dataiter = iter(train_loader)
    inputs = dataiter.next()

    if args.plot:
        # Print elements of dataset
        if args.dataset == 'gc':
            plot_grandcentral(inputs)
        elif args.dataset == 'eth':
            plot_eth(inputs)
        elif args.dataset == 'hotel':
            plot_hotel(inputs)

    # Define optimizer
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam(args.network.parameters(),
                                    lr=args.learning_rate, betas=(.5, .999))
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD(args.network.parameters(),
                                   lr=args.learning_rate, momentum=0.9)

    elif args.optimizer == 'lbfgs':
        args.optimizer = optim.LBFGS(args.network.parameters())

    # Set loss function
    args.criterion = elbo_loss_function

    # restore checkpoint
    restore_checkpoint(args)

    # Set best for minimization
    args.best = float('inf')

    print('Started Training')
    # loop over the dataset multiple times
    for epoch in range(args.epochs):

        # reset running loss statistics
        args.train_loss = args.running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            # Unpack batch
            inputs = batch

            # Send to device
            inputs = inputs.to(args.device)

            # Calculate gradients and update
            with autograd.detect_anomaly():
                # zero the parameter gradients
                args.optimizer.zero_grad()

                # forward + backward + optimize
                outputs, z_mu, z_log_sigma2 = args.network(inputs)
                loss = args.criterion(outputs, inputs, z_mu, z_log_sigma2)
                loss.backward()
                args.optimizer.step()

            # Batch status
            batch_status(batch_idx, inputs, outputs, epoch,
                         train_loader, loss, validset)

        print('====> Epoch: {} '
              'Average loss: {:.4f}'
              .format(epoch, args.train_loss / len(train_loader)))

    # Add trained model
    print('Finished Training')


def validate(validset, print_info=False, log_info=False, global_step=0):
    # Create dataset loader
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=False)
    if print_info:
        print('Started Validation')

    run_loss = 0
    for batch_idx, batch in enumerate(valid_loader, 1):
        # Unpack batch
        inputs, targets = batch

        # Send to device
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Calculate gradients and update
        with autograd.detect_anomaly():
            # forward
            outputs = args.net(inputs)

            # calculate loss
            loss = args.criterion(outputs, targets)
            run_loss += loss.item()

        if batch_idx == 1:
            # Add to tensorboard
            add_tensorboard(inputs, targets, outputs,
                            global_step, name='Valid')

    if log_info:
        args.writer.add_scalar('Valid/loss',
                               run_loss / len(valid_loader),
                               global_step)

    return run_loss / len(valid_loader)


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

    if args.predict:
        # restore checkpoint
        restore_checkpoint(args)

        # Predict test
        generate_samples(trn, args)
    else:
        # Train network
        train(trn, vld)

        # Generate samples
        generate_samples(trn, args)

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
