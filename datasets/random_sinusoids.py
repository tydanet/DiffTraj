import torch
from torch.utils.data import Dataset

from utils.utils import get_conditioning

class RandomSinusoids(Dataset):
    def __init__(self, size, seq_len, channels, return_cond=False, swap_channels=True):
        super().__init__()

        self.size = size
        self.channels = channels
        self.return_cond = return_cond
        self.swap_channels = swap_channels
        self.x = torch.linspace(0, 2 * torch.pi, seq_len)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        amp, freq, phase = torch.randn((3, self.channels, 1))
        y = amp * torch.sin(freq * self.x + phase)

        if self.swap_channels:
            y = y.T

        if self.return_cond:
            if self.swap_channels:
                cond = get_conditioning(y[None, :, :])[0]
                
            else:
                cond = get_conditioning(y.T[None, :, :])[0]

            return y, cond
        
        else:
            return y
