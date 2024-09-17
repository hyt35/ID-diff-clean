import torch
import numpy as np
from torch.utils.data import TensorDataset

class KSphereDataset(TensorDataset):
    def __init__(self, args):
        data_samples = getattr(args, 'data_samples', 10000)
        n_spheres = getattr(args, 'n_spheres', 1)
        ambient_dim = getattr(args, 'ambient_dim', 10)
        manifold_dim = getattr(args, 'manifold_dim', 3)
        noise_std = getattr(args, 'noise_std', 0.05)
        embedding_type = getattr(args, 'embedding_type', 'random_isometry')
        radii = getattr(args, 'radii', [])
        angle_std = getattr(args, 'angle_std', -1)
        
        self.data = self.generate_data(data_samples,
                                       n_spheres,
                                       ambient_dim,
                                       manifold_dim,
                                       noise_std,
                                       embedding_type,
                                       radii,
                                       angle_std)
        
        # Ensure `self.data` is a tensor
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Generated data must be a PyTorch tensor")

        # Initialize the TensorDataset with the generated data
        super(KSphereDataset, self).__init__(self.data)

    def generate_data(self, n_samples, n_spheres, ambient_dim,
                      manifold_dim, noise_std, embedding_type,
                      radii, angle_std):
        if not radii:
            radii = [1] * n_spheres

        if isinstance(manifold_dim, int):
            manifold_dims = [manifold_dim] * n_spheres
        elif isinstance(manifold_dim, list):
            manifold_dims = manifold_dim

        data = []
        for i in range(n_spheres):
            manifold_dim = manifold_dims[i]
            new_data = self.sample_sphere(n_samples, manifold_dim, angle_std)
            new_data = new_data * radii[i]

            if embedding_type == 'random_isometry':
                # random isometric embedding
                randomness_generator = torch.Generator().manual_seed(0)
                embedding_matrix = torch.randn(size=(ambient_dim, manifold_dim+1), generator=randomness_generator)
                q, r = np.linalg.qr(embedding_matrix.numpy())
                q = torch.from_numpy(q)
                new_data = (q @ new_data.T).T
            elif embedding_type == 'first':
                # embedding into first manifold_dim + 1 dimensions
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            elif embedding_type == 'separating':
                # embedding which puts spheres in non-intersecting dimensions
                if n_spheres * (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Cant fit that many spheres. Ensure that n_spheres * (manifold_dim + 1) <= ambient_dim')
                prefix_zeros = torch.zeros((n_samples, i * (manifold_dim + 1)))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            elif embedding_type == 'along_axis':
                # embedding which puts spheres in non-intersecting dimensions
                if (n_spheres - 1) + (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Cant fit that many spheres.')
                prefix_zeros = torch.zeros((n_samples, i))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            else:
                raise RuntimeError('Unknown embedding type.')

            # add noise
            new_data = new_data + noise_std * torch.randn_like(new_data)
            data.append(new_data)

        data = torch.cat(data, dim=0)
        return data

    def sample_sphere(self, n_samples, manifold_dim, std=-1):
        def polar_to_cartesian(angles):
            xs = []
            sin_prod = 1
            for i in range(len(angles)):
                x_i = sin_prod * torch.cos(angles[i])
                xs.append(x_i)
                sin_prod *= torch.sin(angles[i])
            xs.append(sin_prod)
            return torch.stack(xs)[None, ...]

        if std == -1:
            new_data = torch.randn((n_samples, manifold_dim + 1))
            norms = torch.linalg.norm(new_data, dim=1)
            new_data = new_data / norms[:, None]
            return new_data
        else:
            sampled_angles = std * torch.randn((n_samples, manifold_dim))
            return torch.cat([polar_to_cartesian(angles) for angles in sampled_angles], dim=0)

class KSphereScaledDataset(TensorDataset):
    def __init__(self, args):
        data_samples = getattr(args, 'data_samples', 10000)
        n_spheres = getattr(args, 'n_spheres', 1)
        ambient_dim = getattr(args, 'ambient_dim', 10)
        manifold_dim = getattr(args, 'manifold_dim', 3)
        noise_std = getattr(args, 'noise_std', 0.05)
        embedding_type = getattr(args, 'embedding_type', 'random_isometry')
        radii = getattr(args, 'radii', [])
        angle_std = getattr(args, 'angle_std', -1)
        
        self.data = self.generate_data(data_samples,
                                       n_spheres,
                                       ambient_dim,
                                       manifold_dim,
                                       noise_std,
                                       embedding_type,
                                       radii,
                                       angle_std)
        
        # Ensure `self.data` is a tensor
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Generated data must be a PyTorch tensor")

        # Initialize the TensorDataset with the generated data
        super(KSphereScaledDataset, self).__init__(self.data)

    def generate_data(self, n_samples, n_spheres, ambient_dim,
                      manifold_dim, noise_std, embedding_type,
                      radii, angle_std):
        if not radii:
            radii = [1] * n_spheres

        if isinstance(manifold_dim, int):
            manifold_dims = [manifold_dim] * n_spheres
        elif isinstance(manifold_dim, list):
            manifold_dims = manifold_dim

        data = []
        for i in range(n_spheres):
            manifold_dim = manifold_dims[i]
            new_data = self.sample_sphere(n_samples, manifold_dim, angle_std)
            new_data = new_data * radii[i]

            if embedding_type == 'random_isometry':
                # random isometric embedding
                randomness_generator = torch.Generator().manual_seed(0)
                embedding_matrix = torch.randn(size=(ambient_dim, manifold_dim+1), generator=randomness_generator)
                q, r = np.linalg.qr(embedding_matrix.numpy())
                q = torch.from_numpy(q)
                new_data = (q @ new_data.T).T
            elif embedding_type == 'first':
                # embedding into first manifold_dim + 1 dimensions
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            elif embedding_type == 'separating':
                # embedding which puts spheres in non-intersecting dimensions
                if n_spheres * (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Cant fit that many spheres. Ensure that n_spheres * (manifold_dim + 1) <= ambient_dim')
                prefix_zeros = torch.zeros((n_samples, i * (manifold_dim + 1)))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            elif embedding_type == 'along_axis':
                # embedding which puts spheres in non-intersecting dimensions
                if (n_spheres - 1) + (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Cant fit that many spheres.')
                prefix_zeros = torch.zeros((n_samples, i))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)
            else:
                raise RuntimeError('Unknown embedding type.')

            # add noise
            new_data = new_data + noise_std * torch.randn_like(new_data)
            data.append(new_data)

        data = torch.cat(data, dim=0)
        return data

    def sample_sphere(self, n_samples, manifold_dim, std=-1):
        def polar_to_cartesian(angles):
            xs = []
            sin_prod = 1
            for i in range(len(angles)):
                x_i = sin_prod * torch.cos(angles[i])
                xs.append(x_i)
                sin_prod *= torch.sin(angles[i])
            xs.append(sin_prod)
            return torch.stack(xs)[None, ...]

        if std == -1:
            new_data = torch.randn((n_samples, manifold_dim + 1))
            norms = torch.linalg.norm(new_data, dim=1)
            new_data = new_data * np.sqrt(manifold_dim) / norms[:, None]
            return new_data
        else:
            sampled_angles = std * torch.randn((n_samples, manifold_dim))
            return torch.cat([polar_to_cartesian(angles) for angles in sampled_angles], dim=0)