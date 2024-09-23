import torch
import numpy as np
from torch.utils.data import TensorDataset

class SpaghettiDataset(TensorDataset):
    def __init__(self, args):
        data_samples = getattr(args, 'data_samples', 10000)
        ambient_dim = getattr(args, 'ambient_dim', 100)
        noise_std = getattr(args, 'noise_std', 0.05)
        scale = getattr(args, 'scale', 1.)
        
        self.data = self.generate_data(data_samples,
                                       ambient_dim,
                                       noise_std,
                                       scale)
        
        # Ensure `self.data` is a tensor
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Generated data must be a PyTorch tensor")

        # Initialize the TensorDataset with the generated data
        super(SpaghettiDataset, self).__init__(self.data)

    def generate_data(self, n_samples, ambient_dim,
                      noise_std, scale):

        new_data = self.sample_spaghetti(n_samples, ambient_dim)
        new_data = new_data * scale

        # add noise
        new_data = new_data + noise_std * torch.randn_like(new_data)

        return new_data

    def sample_spaghetti(self, n_samples, manifold_dim):
        # (sin x, sin 2x, sin 3x,...)
        multipliers = torch.arange(1,manifold_dim+1)
        angles = torch.rand((n_samples, 1)) * 2 * np.pi
        new_data = torch.sin(angles * multipliers[None,:])
        return new_data

