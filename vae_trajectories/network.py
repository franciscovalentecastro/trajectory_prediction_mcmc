import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, args):
        super(VAE, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_interm_points = args.num_interm_points
        self.condition_dimension = args.condition_dimension

        # Encoder
        self.encoder_1 = nn.Linear(2 * self.num_interm_points +
                                   2 * self.condition_dimension,
                                   args.hidden_size)
        self.encoder_2 = nn.Linear(args.hidden_size, args.hidden_size)

        self.z_mu = nn.Linear(args.hidden_size, args.latent_dim)
        self.z_log_sigma2 = nn.Linear(args.hidden_size, args.latent_dim)

        # Decoder
        self.decoder_1 = nn.Linear(args.latent_dim +
                                   2 * self.condition_dimension,
                                   args.hidden_size)
        self.decoder_2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.decoder_3 = nn.Linear(args.hidden_size,
                                   2 * self.num_interm_points)

    def encode(self, x):
        x = torch.tanh(self.encoder_1(x))
        x = torch.tanh(self.encoder_2(x))

        return self.z_mu(x), self.z_log_sigma2(x)

    def sample(self, z_mu, z_log_sigma2):
        z_std = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_std)

        return z_mu + epsilon * z_std

    def decode(self, z):
        h1 = torch.tanh(self.decoder_1(z))
        h1 = torch.tanh(self.decoder_2(h1))
        h2 = self.decoder_3(h1)

        return h2

    def forward(self, x):
        # Start and goal points
        start = x[:, 0].flatten(start_dim=1)
        mid = x[:, 1:-1].flatten(start_dim=1)
        goal = x[:, -1].flatten(start_dim=1)

        # Check condition dimension
        if self.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Add conditional variables
        x_cat = torch.cat([mid, start, goal], dim=1)

        z_mu, z_log_sigma2 = self.encode(x_cat)
        z = self.sample(z_mu, z_log_sigma2)

        # Add conditional variables
        z_cat = torch.cat([z, start, goal], dim=1)

        # Decode and concatenate
        decoded_z = self.decode(z_cat)

        # Check condition dimension
        if self.condition_dimension == 1:
            start = x[:, 0].flatten(start_dim=1)
            goal = x[:, -1].flatten(start_dim=1)

        # Construct tensor
        result = torch.cat([start, decoded_z, goal], dim=1)
        result = result.view(self.batch_size,
                             self.num_interm_points + 2, 2)

        return result, z_mu, z_log_sigma2


class VAE_DELTA(nn.Module):

    def __init__(self, args):
        super(VAE_DELTA, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_interm_points = args.num_interm_points
        self.condition_dimension = args.condition_dimension

        # Encoder
        self.encoder_1 = nn.Linear(2 * self.num_interm_points +
                                   2 * self.condition_dimension,
                                   args.hidden_size)
        self.encoder_2 = nn.Linear(args.hidden_size, args.hidden_size)

        self.z_mu = nn.Linear(args.hidden_size, args.latent_dim)
        self.z_log_sigma2 = nn.Linear(args.hidden_size, args.latent_dim)

        # Decoder
        self.decoder_1 = nn.Linear(args.latent_dim +
                                   2 * self.condition_dimension,
                                   args.hidden_size)
        self.decoder_2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.decoder_3 = nn.Linear(args.hidden_size,
                                   2 * self.num_interm_points)

    def encode(self, x):
        x = torch.tanh(self.encoder_1(x))
        x = torch.tanh(self.encoder_2(x))

        return self.z_mu(x), self.z_log_sigma2(x)

    def sample(self, z_mu, z_log_sigma2):
        z_std = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_std)

        return z_mu + epsilon * z_std

    def decode(self, z):
        h1 = torch.tanh(self.decoder_1(z))
        h1 = torch.tanh(self.decoder_2(h1))
        h2 = self.decoder_3(h1)

        return h2

    def forward(self, x):
        # Start and goal points
        start = x[:, 0].flatten(start_dim=1)
        goal = x[:, -1].flatten(start_dim=1)

        # Delta of consecutive coordinates
        delta = (x[:, 1:-1] - x[:, :-2]).flatten(start_dim=1)

        # Check condition dimension
        if self.condition_dimension == 1:
            start = get_angle(start)
            goal = get_angle(goal)

        # Add conditional variables
        x_cat = torch.cat([delta, start, goal], dim=1)

        z_mu, z_log_sigma2 = self.encode(x_cat)
        z = self.sample(z_mu, z_log_sigma2)

        # Add conditional variables
        z_cat = torch.cat([z, start, goal], dim=1)

        # Decode and concatenate
        decoded_z = self.decode(z_cat)

        # Check condition dimension
        if self.condition_dimension == 1:
            start = x[:, 0].flatten(start_dim=1)
            goal = x[:, -1].flatten(start_dim=1)

        # Construct tensor
        result = torch.cat([start, decoded_z, goal], dim=1)
        result = result.view(self.batch_size,
                             self.num_interm_points + 2, 2)

        # From delta get coordinates
        for idx in range(1, self.num_interm_points + 1):
            result[:, idx, :] += result[:, idx - 1, :]

        return result, z_mu, z_log_sigma2


def elbo_loss_function(decoded_x, x, z_mu, z_log_sigma2):
    # Exctract only important dimension
    x = x.flatten(start_dim=1)
    decoded_x = decoded_x.flatten(start_dim=1)

    # log P(x|z) per element in batch
    # MSE because it is the logarithm of a Gaussian
    LOGP = F.mse_loss(decoded_x, x, reduction='none')
    LOGP = torch.sum(LOGP, dim=1)

    # DKL(Q(z|x)||P(z)) per element in batch
    DKL = 0.5 * torch.sum(z_log_sigma2.exp() +
                          z_mu.pow(2) -
                          1.0 -
                          z_log_sigma2,
                          dim=1)

    # Average loss in batch
    return torch.mean(LOGP + DKL)


def get_angle(x):
    # Necessary values
    batch_size = x.shape[0]

    # Point in the middle of image
    center = torch.tensor([[1920.0 / 2.0, 1080.0 / 2.0]],
                          dtype=torch.float32).repeat(batch_size, 1)
    right = torch.tensor([[1920.0, 1080.0 / 2.0]],
                         dtype=torch.float32).repeat(batch_size, 1)

    # Get angle between vectors
    a = right - center
    b = x - center

    # Normalize vector
    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)

    # Get angle in radians
    angle = torch.atan2(b[:, 0], b[:, 1]) - torch.atan2(a[:, 0], a[:, 1])
    angle = angle.reshape(-1, 1)

    # Convert to degrees
    return (angle * 180 / math.pi) % 360.0


def project_to_boundary(points):
    # Boundary points
    x = [450, 1420, 1900, 35, 450]
    y = [170, 170, 1000, 1000, 170]

    # Point in the middle of the image
    c_x = 1920.0 / 2.0
    c_y = 1080.0 / 2.0
    center = torch.tensor([c_x, c_y], dtype=torch.float32)

    # Required
    epsilon = 1e-2

    # Tensor to save transform
    tPoints = torch.zeros_like(points)

    # For each element in batch
    for kdx in range(points.shape[0]):
        # Construct vector
        vec = points[kdx, :] - center

        # Intersect with each side
        for idx in range(4):
            # Get coordinates
            x_1 = x[idx]
            y_1 = y[idx]
            x_2 = x[idx + 1]
            y_2 = y[idx + 1]

            # Line equation
            a = (y_2 - y_1) / (x_2 - x_1)
            b = y_1 - a * x_1

            # Get intersection
            t = (-c_y + a * c_x + b) / (vec[1] - a * vec[0])
            inter = t * vec + center

            # Check if interesected
            if(t > 0 and
               inter[0] >= min(x_1, x_2) - epsilon and
               inter[0] <= max(x_1, x_2) + epsilon and
               inter[1] >= min(y_1, y_2) - epsilon and
               inter[1] <= max(y_1, y_2) + epsilon):

                tPoints[kdx, :] = inter

    return tPoints
