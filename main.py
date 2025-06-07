import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from numpy import dtype
from pandas import json_normalize
from torch.utils.data import TensorDataset, DataLoader

# from utils.config import args
from datasets.random_sinusoids import RandomSinusoids
from utils import utils
from utils.EMA import EMAHelper
from utils.Traj_UNet import *

# This code part from https://github.com/sunlin-ai/diffusion_tutorial

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def main(config):
    
    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNetContinuous(config).to(config.training.device)

    if config.data.dataset == 'Path':
        traj = np.load(config.data.datasets.path.traj_path1, allow_pickle=True)
        head = np.load(config.data.head_path2, allow_pickle=True)
        assert traj.shape[1] == config.data.traj_length
        assert traj.shape[2] == config.data.channels
        assert head.shape[1] == config.model.attr_dim

    elif config.data.dataset == 'RandomSinusoids':
        tmp_dataset = RandomSinusoids(
            1, 
            config.data.traj_length,
            config.data.channels)
        n = config.data.datasets.random_sinusoids.warmup_size
        traj = torch.stack([tmp_dataset[0] for _ in range(n)])
        head = utils.get_conditioning(traj)

    tmu = traj.mean(axis=0, keepdims=1)
    tsigma = traj.std(axis=0, keepdims=1)
    traj = (traj - tmu) / tsigma
    np.save('/tmp/traj_mu.npy', tmu)
    np.save('/tmp/traj_sigma.npy', tsigma)
    mlflow.log_artifact('/tmp/traj_mu.npy')
    mlflow.log_artifact('/tmp/traj_sigma.npy')

    hmu = head.mean(axis=0, keepdims=1)
    hsigma = head.std(axis=0, keepdims=1)
    head = (head - hmu) / hsigma
    np.save('/tmp/head_mu.npy', hmu)
    np.save('/tmp/head_sigma.npy', hsigma)
    mlflow.log_artifact('/tmp/head_mu.npy')
    mlflow.log_artifact('/tmp/head_sigma.npy')

    traj = np.swapaxes(traj, 1, 2)
    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()

    input_schema = Schema([
        TensorSpec(dtype('float32'), (-1, config.data.channels, config.data.traj_length), 'x'),
        TensorSpec(dtype('float32'), (-1,), 't'),
        TensorSpec(dtype('float32'), (-1, head.shape[1]), 'attr'),
    ])
    output_schema = Schema([TensorSpec(dtype('float32'), (-1, config.data.channels, config.data.traj_length))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    ###########################################################
    # The input shape of traj and head list as follows:
    # traj: [batch_size, 2, traj_length]   2: latitude and longitude
    # head: [batch_size, 8]   8: departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed, start_id, end_id
    ###########################################################
    if config.data.dataset == 'Path':
        dataset = TensorDataset(traj, head)
    
    elif config.data.dataset == 'RandomSinusoids':
        dataset = RandomSinusoids(
            config.data.datasets.random_sinusoids.size, 
            config.data.traj_length, 
            config.data.channels)
        
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=config.training.num_workers)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(
        config.diffusion.beta_start,
        config.diffusion.beta_end, 
        n_steps).to(config.training.device)
    
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = float(config.training.lr)  # Explore this - might want it lower when training on the full dataset

    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    for epoch in tqdm.tqdm(range(1, config.training.n_epochs + 1)):
        losses = []

        for _, (trainx, head) in enumerate(dataloader):
            x0 = trainx.to(config.training.device)
            head = head.to(config.training.device)
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1,)).to(config.training.device)
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head)
            # Compare the predictions with the targets
            loss = config.training.reconstruction_loss_w * F.mse_loss(noise.float(), pred_noise)

            if config.training.condition_loss_w > 0:
                noise_cond = utils.get_conditioning(noise.swapaxes(1, 2))
                pred_noise_cond = utils.get_conditioning(pred_noise.swapaxes(1, 2))
                cond_loss = config.training.condition_loss_w * F.mse_loss(noise_cond, pred_noise_cond)
                loss = loss + cond_loss

            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

            if config.model.ema:
                ema_helper.update(unet)

        if epoch % config.training.log_interval == 0:
            mlflow.pytorch.log_model(unet, f'unet_{epoch:05}', signature=signature)

        mlflow.log_metric('Loss', f"{np.mean(losses):4f}", step=epoch)
        mlflow.pytorch.log_model(unet, f'unet_latest', signature=signature)

if __name__ == "__main__":
    import argparse

    import yaml

    with mlflow.start_run():
        mlflow.set_tracking_uri('http://127.0.0.1:9090')

        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', required=True)
        args = parser.parse_args()
        mlflow.log_artifact(args.config)

        with open(args.config, 'r') as f:
            args = yaml.load(f, Loader=yaml.CLoader)

        params = json_normalize(args).T.to_dict().get(0)
        mlflow.log_params(params)
        config = utils.load_config(args)

        try:
            main(config)

        except KeyboardInterrupt:
            pass
