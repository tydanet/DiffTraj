import torch
from torch.utils.data import Dataset

class RandomSinusoids(Dataset):
    def __init__(self, size, seq_len, channels):
        super().__init__()

        self.size = size
        self.channels = channels
        self.x = torch.linspace(0, 2 * torch.pi, seq_len)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        amp, freq, phase = torch.randn((3, self.channels, 1))
        y = amp * torch.sin(freq * self.x + phase)
        return y.T
