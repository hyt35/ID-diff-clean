import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os

from data.data_utils import get_dataloaders
from models.mlp import MLP
from models.unet import UNet
from utils.sde import VESDE
from utils.train_utils import EMA, WarmUpScheduler, save_model, get_score_fn, eval_callback
from utils.sampling_utils import generation_callback
from torch.distributions import Uniform
import json

def get_DSM_loss_fn(sde, likelihood_weighting):
  def loss_fn(score_fn, batch, t_dist):
      x = batch
      t = t_dist.sample((x.shape[0],)).type_as(x)
      n = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
      
      score = score_fn(perturbed_x, t)
      grad_log_pert_kernel = -1 * n / std[(...,) + (None,) * len(x.shape[1:])]
      losses = torch.square(score - grad_log_pert_kernel)
            
      if likelihood_weighting:
        _, g = sde.sde(torch.zeros_like(x), t, True)
        w2 = g ** 2
      else:
        w2 = std ** 2
            
      importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
      losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
      losses *= 1/2
      loss = torch.mean(losses)
      return loss
  return loss_fn

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

def train(args):
    # Set up logging directories
    tensorboard_dir = os.path.join(args.base_log_dir, args.experiment, 'training_logs')
    checkpoint_dir = os.path.join(args.base_log_dir, args.experiment, 'checkpoints')
    eval_dir = os.path.join(args.base_log_dir, args.experiment, 'eval')
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.network == 'MLP':
        model = MLP(args).to(device)
    elif args.network == 'U-NET':
        model = UNet(args).to(device)

    print_model_size(model)  # Print the size of the model

    sde = VESDE()
    sde.sampling_eps = 1e-5  # or use a value from args if needed
    ema_model = EMA(model=model, decay=0.999)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = WarmUpScheduler(optimizer, args.learning_rate, warmup_steps=1000)

    step = 0
    best_checkpoints = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    t_dist = Uniform(sde.sampling_eps, 1)
    loss_fn = get_DSM_loss_fn(sde, likelihood_weighting=False)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()
            x = data[0].to(device)

            score_fn = get_score_fn(sde, model)
            loss = loss_fn(score_fn, x, t_dist)


            loss.backward()
            optimizer.step()
            ema_model.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        ema_model.apply_shadow()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{args.epochs}"):
                x = data[0].to(device)

                score_fn = get_score_fn(sde, model)
                loss = loss_fn(score_fn, x, t_dist)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        if (epoch + 1) % args.vis_frequency == 0:
            steps = args.steps
            num_samples = args.num_samples
            shape = (num_samples, args.ambient_dim)
            generation_callback(writer, sde, model, steps, shape, device, epoch)
            
        ema_model.restore()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience_epochs:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_model(model, ema_model, epoch, val_loss, "Model", checkpoint_dir, best_checkpoints)


    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for Diffusion/Score Model")

    # Logging settings
    parser.add_argument("--base_log_dir", type=str, default="./results", help="Base directory for logs and checkpoints.")
    parser.add_argument("--experiment", type=str, default="deep_sphere_2", help="Experiment name for directory structure.")

    # Data settings
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--dataset", type=str, choices=['sphere', 'euclidean', 'mnist'], default='sphere', help="Type of data.")
    parser.add_argument("--data_samples", type=int, default=10000, help="Number of samples per sphere")
    parser.add_argument("--n_spheres", type=int, default=1, help="Number of spheres")
    parser.add_argument("--ambient_dim", type=int, default=2, help="Dimension of the ambient space")
    parser.add_argument("--manifold_dim", type=int, default=1, help='Manifold dimension')
    parser.add_argument("--noise_std", type=float, default=0.0, help='Standard deviation of the noise')
    parser.add_argument("--embedding_type", type=str, default='random_isometry', help='Type of embedding')
    parser.add_argument("--radii", nargs='*', type=float, help='Radii of the spheres')
    parser.add_argument("--angle_std", type=float, default=-1, help='Standard deviation of angles for sampling')

    # Model settings
    parser.add_argument('--network', type=str, choices=['MLP', 'U-NET'], default='MLP', help='Neural Network type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for MLP/FCN')
    parser.add_argument('--depth', type=int, default=5, help='Depth (number of hidden layers) for MLP/FCN')
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs.")
    parser.add_argument("--checkpoint_frequency", type=int, default=20, help="Frequency of saving checkpoints.")
    parser.add_argument("--patience_epochs", type=int, default=100, help="Number of epochs to wait for improvement before early stopping.")

    # Device settings
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (e.g., 'cuda' or 'cpu').")

    # Evaluation settings
    parser.add_argument("--eval_callback_epochs", type=int, default=20, help="Frequency of evaluation callback execution.")
    parser.add_argument("--num_eval_points", type=int, default=10, help="Number of points for evaluating the manifold dimension.")
    parser.add_argument("--eval_save_path", type=str, default="./eval", help="Directory to save evaluation results.")

    # Visualization settings
    parser.add_argument("--vis_frequency", type=int, default=50, help="Frequency of visualization during training.")
    parser.add_argument("--steps", type=int, default=1024, help="Number of steps for sample generation.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate.")

    args = parser.parse_args()

    # Save arguments to a file
    args_dict = vars(args)
    os.makedirs(os.path.join(args.base_log_dir, args.experiment), exist_ok=True)
    with open(os.path.join(args.base_log_dir, args.experiment, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    train(args)
