import numpy as np
import torch
from typing import Literal, Union

def retrieve_n_dims(svals, mode: Union[Literal["subtract", "divide"], float]) -> int:
    svals = torch.Tensor(svals)
    if mode == 'subtract':
        diffs = svals[:-1] - svals[1:]
        return len(svals) - torch.argmax(diffs) - 1

    elif mode == 'divide':
        diffs = svals[:-1] / svals[1:]
        return len(svals) - torch.argmax(diffs) - 1
    
    elif isinstance(mode, float):
        diffs = svals[:-1] / svals[1:]
        return len(svals) - torch.nonzero(diffs > mode, as_tuple = True)[0][-1] - 1
    else:
        raise Exception(f"mode {mode} is not defined")
def dist_to_tangent(p, q, sv_p, dim_intrinsic):

    # Approximately computes the distance of point q to the tangent plane T_p M
    # allow batching
    # assumes singular vectors are ordered []
    if q.dim() == 1:
        q = torch.unsqueeze(q,0)
    if q.dim() != 2:
        raise Exception("q must have shape [N,d] or [d]")
    if p.dim() != 1:
        raise Exception("p must have shape [d]")

    vec = q - p # [N,d]
    
    # compute normal subroutine
    tangent_vecs = sv_p[-dim_intrinsic:] # [dim_intrinsic, d],
    # compute dot product
    dist_arr = vec @ tangent_vecs.T # [N, dim_intrinsic]
    norm_vec = torch.linalg.norm(vec, dim=1)
    norm_tangent = torch.linalg.norm(dist_arr, dim = 1) # reduce over ambient dimension, [N]
    return torch.sqrt(norm_vec ** 2 - norm_tangent ** 2) # vector of length [N] or [1]