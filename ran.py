#%% 1-sphere in R^2
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_2 --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=4. --ambient_dim=2 --manifold_dim=1

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_2

#%% 1-sphere in R^5
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_1_5 --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=4. --ambient_dim=5 --manifold_dim=1

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_1_5

#%% 2-sphere in R^5
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_2_5 --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=4. --ambient_dim=5 --manifold_dim=2

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_2_5
#%% 10-sphere in R^100
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_10 --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=4. --ambient_dim=100 --manifold_dim=10

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_10

#%% 10-sphere in R^100
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_10_LR --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=4. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_10_LR

#%% 50-sphere in R^100
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_50_LR --device=cuda --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=4. --ambient_dim=100 --manifold_dim=50 --patience_epochs=200

CUDA_VISIBLE_DEVICES=2 python eval.py --experiment=deep_sphere_50_LR

#%% 10-sphere scaled
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_10_LR --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=8. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200

#%% 50-sphere scaled
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_50_LR --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=50 --patience_epochs=200


CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_10 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200

CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_50 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=50 --patience_epochs=200


#%% 17-07-24
#* Scaled Spheres
# Still has 11 small SV
# Poor norm concentration
# 176 1.4 1.2: 11 eigenvalues <= 1.4
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_10_LR2 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200

# 5000 epochs with small LR: has 10 small SV
# very good concentration
# 185 73.9 4.5 ... : 10 eigenvalues <= 4.5
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_scaled_10 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=1500 --epochs=5000

#%% scaled and first
# 11, with not-so-visible jump
# ... 188 6.5 4.5 4.0 ... : 10 eigenvalues <= 4.5
# trained for 1000 iters, but loss is still decreasing
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_s_f_10 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200 --embedding_type=first

# ... 393 31 7.6 7.4 ... : 50 eigenvalues <= 7.6 (approx correct)
# decent concentration, but still decreasing
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_s_f_50 --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=50 --patience_epochs=200 --embedding_type=first

# 171 6.3 1.4 : 10 eigenvalues <= 1.4
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_s_f_10_LR --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=200 --embedding_type=first

# 385 12 8.7 8.1 ... : 50 eigenvalues <= 8.7
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_s_f_50_LR --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-4 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=50 --patience_epochs=200 --embedding_type=first