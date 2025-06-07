from math import sin, cos, sqrt, radians, asin
from types import SimpleNamespace

import numpy as np
import torch

def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def time_warping(x, length=200):
    """
    Resamples a trajectory to a new length.
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    warped_trajectory = np.zeros((2, length))
    for i in range(2):
        warped_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return warped_trajectory.T

def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def q_xt_x0(x0, t, alpha_bar):
    # get mean and var of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also returns noise

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.to(xt.device.type), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next

def divide_grids(boundary, grids_num):
    lati_min, lati_max = boundary['lati_min'], boundary['lati_max']
    long_min, long_max = boundary['long_min'], boundary['long_max']
    # Divide the latitude and longitude into grids_num intervals.
    lati_interval = (lati_max - lati_min) / grids_num
    long_interval = (long_max - long_min) / grids_num
    # Create arrays of latitude and longitude values.
    latgrids = np.arange(lati_min, lati_max, lati_interval)
    longrids = np.arange(long_min, long_max, long_interval)
    return latgrids, longrids

# calculte the distance between two points
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def load_config(args):
    temp = {}

    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)

    config = SimpleNamespace(**temp)
    return config

def run_diffusion(net, head, config, device=None):
    if device is None:
        device = config.training.device

    beta = torch.linspace(
        config.diffusion.beta_start, 
        config.diffusion.beta_end, 
        config.diffusion.num_diffusion_timesteps).to(device)
    
    num_samples = head.shape[0]
    seq = range(0, config.diffusion.num_diffusion_timesteps)

    with torch.no_grad():
        X = torch.randn(num_samples, config.data.channels, config.data.traj_length).to(device)
        seq_next = [-1] + list(seq[:-1])

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((num_samples,), i, device=device)
            next_t = torch.full((num_samples,), j, device=device)
            pred_noise = net(X, t, head)
            X = p_xt(X, pred_noise, t, next_t, beta)

    return X

def get_conditioning(x):
    origin = x[:, 0]
    destination = x[:, -1]

    deltas = x.diff(dim=1)
    dists = torch.norm(deltas, dim=2)

    mean_step_size = dists.mean(dim=1)
    step_size_dev = dists.std(dim=1)
    max_step_size = dists.max(dim=1).values
    length = dists.sum(dim=1)

    center_of_mass = x.mean(dim=1)
    
    bottom_left = x.min(dim=1).values
    top_right = x.max(dim=1).values

    cond = torch.cat((
        origin, 
        destination, 
        center_of_mass,
        bottom_left,
        top_right,
        mean_step_size.unsqueeze(1), 
        step_size_dev.unsqueeze(1),
        max_step_size.unsqueeze(1),
        length.unsqueeze(1),
    ), dim=1)

    return cond
