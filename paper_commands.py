
# Visualising the 
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=deep_sphere_s_f_10_LR --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=14. --ambient_dim=100 --manifold_dim=10 --patience_epochs=1500 --epochs=5000

python eval_save_full_SVD.py --experiment=deep_sphere_s_f_10_LR --force_first=10

# Spaghetti experiments
CUDA_VISIBLE_DEVICES=2 python train.py --experiment=spaghetti --device=cuda --dataset=sphere_scaled --network=MLP --hidden_dim=2048 --learning_rate=2e-5 --ema=0.9999 --vesde_max=4. --ambient_dim=100 --patience_epochs=1500 --epochs=5000