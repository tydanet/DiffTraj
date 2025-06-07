import mlflow
import numpy as np
import plotly.graph_objects as go
import torch
import yaml

from utils import utils

device = 'mps'
runid = 'bcc1ff6bd42745e9b7c4c6c3254825a4'
model_uri = f'runs:/{runid}/unet_latest'
mlflow.set_tracking_uri('http://127.0.0.1:9090')
net = mlflow.pytorch.load_model(model_uri).to(device)
net.eval()
# X = np.load('./data/randsine/X.npy')

# X_head = np.load('./data/randsine/X_head.npy')

# hmu_path = mlflow.artifacts.download_artifacts(f'runs:/{runid}/head_mu.npy')
# hmu = np.load(hmu_path)

# hsigma_path = mlflow.artifacts.download_artifacts(f'runs:/{runid}/head_sigma.npy')
# hsigma = np.load(hsigma_path)

# X_head = torch.FloatTensor((X_head - hmu) / hsigma).to(device)

# head = torch.tile(X_head[[0]], (10, 1))
torch.manual_seed(0)
head = torch.randn((10, 14)).to(device)
# force all to have same od pair
head[:, 0:4] = head[0, 0:4]

with open('./config.yml', 'r') as f:
    args = yaml.load(f, Loader=yaml.CLoader)

config = utils.load_config(args)
samples = utils.run_diffusion(net, head, config, device).cpu()

tmu_path = mlflow.artifacts.download_artifacts(f'runs:/{runid}/traj_mu.npy')
tmu = np.load(tmu_path).transpose((0, 2, 1))

tsigma_path = mlflow.artifacts.download_artifacts(f'runs:/{runid}/traj_sigma.npy')
tsigma = np.load(tsigma_path).transpose((0, 2, 1))

samples = samples * tsigma + tmu

fig = go.Figure()

# for Xi in X[[0]]:
#     x, y = Xi.T
#     fig.add_trace(go.Scatter(x=x, y=y))
head = head.cpu()
fig.add_trace(go.Scatter(x=head[0, [0, 2]], y=head[0, [1, 3]], mode='markers'))

for x, y in samples:
    fig.add_trace(go.Scatter(x=x, y=y))

fig.show()
