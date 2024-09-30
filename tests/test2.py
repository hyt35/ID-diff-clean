import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )


from utils.svd_utils import retrieve_n_dims, dist_to_tangent
import torch
import numpy as np
import pickle

print('''### Beginning Test 2 ###
      Compute approximate tangent distance using estimator
      ''')


eval_dir = os.path.join('results', "deep_sphere_s_f_10_LR", 'eval_T')
with open(os.path.join(eval_dir, 'svd.pkl'), 'rb') as f:
    data = pickle.load(f)
singular_values = data['singular_values']
singular_vectors = np.load(os.path.join(eval_dir, 'sv.npy')) # [N_points, ambient_dim, ambient_dim]

print(f"Expected {np.sqrt(10)*2}")

p = torch.zeros(100)
p[0] = np.sqrt(10)

q = torch.zeros(100)
q[0] = -np.sqrt(10)

print("ev1, dim=10, dist(q-p, T_p M)", dist_to_tangent(p, q, sv_p = singular_vectors[0], dim_intrinsic = 10), "-> almost correct")
print("ev1, dim=11, dist(q-p, T_p M)", dist_to_tangent(p, q, sv_p = singular_vectors[0], dim_intrinsic = 11), "-> almost contained in tangent space")

p = torch.zeros(100)
p[1] = np.sqrt(10)

q = torch.zeros(100)
q[1] = -np.sqrt(10)

print("ev2, dim=10", dist_to_tangent(p, q, sv_p = singular_vectors[1], dim_intrinsic = 10))
