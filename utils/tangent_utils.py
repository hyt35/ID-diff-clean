import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import os
import numpy as np
import abc
from contextlib import contextmanager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eval_callback_fullSVD(score_fn, sde, val_dataloader, num_datapoints, device, save_path, name=None, return_svd=False, force_first = -1):
    #* Compute and save SVD decomposition
    #* eval_callback saves only singular values
    #* This one saves the 
    # EG: input [64,100] matrix
    # SVD produces M = U D V, M is [500,100], V is [100,100], D is [500,100] with s.v. on diag
    # eval_callback saves D, we save V as well
    os.makedirs(save_path, exist_ok=True)

    singular_values = []
    singular_vectors = []
    idx = 0
    sampling_eps = sde.sampling_eps

    with tqdm(total=num_datapoints) as pbar:
        for x in val_dataloader:
            # print(len(x))
            orig_batch = x[0]
            orig_batch = orig_batch.to(device)
            batchsize = orig_batch.size(0)

            if idx >= num_datapoints:
                break

            for x in orig_batch:
                if idx >= num_datapoints:
                    break
                # Override first
                if force_first != -1:
                    x = torch.zeros_like(x)
                    x[idx] = np.sqrt(force_first)
                ambient_dim = np.prod(x.shape[1:])
                x = x.repeat([batchsize] + [1 for _ in range(len(x.shape))])
                # print("Repeated x shape:", x.shape)

                num_batches = int(np.floor(ambient_dim / batchsize)) + 1
                num_batches *= 2

                t = sampling_eps
                vec_t = torch.ones(x.size(0), device=device) * t

                scores = []
                for i in range(1, num_batches + 1):
                    batch = x.clone()

                    mean, std = sde.marginal_prob(batch, vec_t)
                    z = torch.randn_like(batch)
                    batch = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
                    # print(batch.size(), vec_t.size())
                    score = score_fn(batch, vec_t).detach().cpu()
                    # print("Score shape:", score.shape)
                    scores.append(score)

                scores = torch.cat(scores, dim=0)
                # print(score.size())
                scores = torch.flatten(scores, start_dim=1)
                # print("Flattened scores shape:", scores.shape)

                means = scores.mean(dim=0, keepdim=True)
                normalized_scores = scores - means

                with torch.no_grad():
                    u, s, v = torch.linalg.svd(normalized_scores)
                singular_values.append(s.tolist())
                singular_vectors.append(v.detach().cpu().numpy())
                # print("Singular values shape:", s.shape)

                idx += 1
                pbar.update(1)

    info = {'singular_values': singular_values}
    if return_svd:
        return info, singular_vectors
    else:
        if name is None:
            name = 'svd'
        with open(os.path.join(save_path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(info, f)
        np.save(os.path.join(save_path, 'sv.npy'), np.asarray(singular_vectors))