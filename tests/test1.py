import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )


from utils.svd_utils import retrieve_n_dims, dist_to_tangent
import torch
import numpy as np
import pickle

print('''### Beginning Test 1 ###
      Compute dimensions for 10-sphere using 3 different dimension estimator
      ''')


eval_dir = os.path.join('results', "deep_sphere_s_f_10_LR", 'eval_T')
with open(os.path.join(eval_dir, 'svd.pkl'), 'rb') as f:
    data = pickle.load(f)
singular_values = data['singular_values']
singular_vectors = np.load(os.path.join(eval_dir, 'sv.npy')) # [N_points, ambient_dim, ambient_dim]


print(retrieve_n_dims(singular_values[0], "subtract"))
print(retrieve_n_dims(singular_values[0], "divide"))
print(retrieve_n_dims(singular_values[0], 1.5))
# print(singular_vectors[0,89])

