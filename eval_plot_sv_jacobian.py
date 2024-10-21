# python eval_plot_sv.py --experiment=deep_sphere_s_f_10_LR --force_first=1
import torch
import argparse
import os
import glob

from data.data_utils import get_dataloaders
from models.mlp import MLP
from models.unet import UNet
from utils.sde import VESDE
from utils.train_utils import EMA, load_model #, eval_callback
from utils.tangent_utils import eval_callback_jacobianSVD
from utils.sampling_utils import get_score_fn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def visualize_data(val_loader, eval_dir):
    points_list = []

    # Iterate through the val_loader to collect points
    for data in val_loader:
        points = data[0]
        points_list.append(points)
        if len(torch.cat(points_list)) >= 1000:
            break
    
    # Concatenate all collected points
    points = torch.cat(points_list)[:1000].cpu().numpy()
    
    # Plot the first 1000 points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('First 1000 Points in Validation Set')
    
    # Save the plot instead of showing it
    plot_path = os.path.join(eval_dir, 'validation_data_plot.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f'Saved validation data plot to {plot_path}')

def evaluate(args):
    # Set up logging directories
    checkpoint_dir = os.path.join(args.base_log_dir, args.experiment, 'checkpoints')
    eval_dir = os.path.join(args.base_log_dir, args.experiment, 'eval_T_jac')
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device(args.device)

    _, val_loader, _ = get_dataloaders(args)

    # Visualize the validation dataset
    #visualize_data(val_loader, eval_dir)

    if args.network == 'MLP':
        model = MLP(args).to(device)
    elif args.network == 'U-NET':
        model = UNet(args).to(device)

    sde = VESDE(args.vesde_min, args.vesde_max)
    sde.sampling_eps = 1e-5  # or use a value from args if needed
    ema_model = EMA(model=model, decay=args.ema)

    # Infer the last EMA checkpoint based on the naming convention
    last_ema_checkpoint_path = glob.glob(os.path.join(checkpoint_dir, "*_last_EMA.pth"))
    if not last_ema_checkpoint_path:
        raise FileNotFoundError("No checkpoint file ending with '_last_EMA.pth' found.")
    last_ema_checkpoint_path = last_ema_checkpoint_path[0]

    load_model(model, ema_model, last_ema_checkpoint_path, "Model", is_ema=True)

    # Apply EMA weights to the model
    ema_model.apply_shadow()

    # Set the model to evaluation mode
    model.eval()

    # Run evaluation callback
    score_fn = get_score_fn(sde, model)  # Use the model with EMA weights applied
    eval_callback_jacobianSVD(score_fn, sde, val_loader, args.num_eval_points, args.device, eval_dir, force_first=args.force_first)

    # Evaluation and plotting
    singular_values_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('svd.pkl')]

    plt.figure(figsize=(10, 6))
    for file in singular_values_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            singular_values = data['singular_values']
            for sv in singular_values:
                plt.plot(sv, alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Spectrum of Singular Values')
    
    # Save the plot instead of showing it
    spectrum_plot_path = os.path.join(eval_dir, 'singular_values_spectrum.png')
    plt.savefig(spectrum_plot_path)
    plt.close()
    print(f'Saved singular values spectrum plot to {spectrum_plot_path}')

    #**********
    #* 2x2 format
    #**********
    # singular_vectors = np.load(os.path.join(eval_dir, 'sv.npy'))
    # fig, ax = plt.subplots(2,2,figsize=(8,8),dpi=300)
    # for i in range(2):
    #     for j in range(2):
    #         ctr = 2*i + j
    #         im = ax[i,j].imshow(singular_vectors[ctr],interpolation='none')
    #         plt.colorbar(im, ax=ax[i,j],fraction=0.046, pad=0.04)
    #         ax[i,j].set_axis_off()
    #         ax[i,j].invert_yaxis()

    #**********
    #* 1x4 format
    #**********
    singular_vectors = np.load(os.path.join(eval_dir, 'sv.npy'))
    fig, ax = plt.subplots(1,4,figsize=(12,3),dpi=300)
    for i in range(4):
        ctr = i
        tmp = singular_vectors[ctr]* np.sign(singular_vectors[ctr][89,i])
        # im = ax[i].imshow(tmp * np.sign(singular_vectors[ctr][89,i]),interpolation='none')
        # singular_vectors[ctr]
        im = ax[i].imshow(tmp,interpolation='none',origin='lower')
        plt.colorbar(im, ax=ax[i],fraction=0.046, pad=0.04,orientation='horizontal')
        ax[i].set_axis_off()
        # ax[i].invert_yaxis()
        axins = zoomed_inset_axes(ax[i], 3, loc=1) # zoom = 6
        # axins.imshow(singular_vectors[ctr], interpolation="none",
        #             origin="lower")

        axins.imshow(tmp, interpolation="none",origin='lower')
        axins.set_xlim(-0.5,11.5)
        axins.set_ylim(87.5,99.5)
        axins.set_xticks([0,9],[1,10])
        axins.set_yticks([90,99],[10,1])
        # for axis in ['top','bottom','left','right']:
        #     # axins.spines[axis].set_linewidth(3)
        #     axins.spines[axis].set_color('r')
        # axins.invert_yaxis()
        # axins.set_axis()
        mark_inset(ax[i], axins, loc1=3, loc2=1, fc="none", ec="r",lw=1)
        # sub region of the original image
        # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)

    sv_plot_path = os.path.join(eval_dir, 'singular_vectors.png')
    fig.savefig(sv_plot_path)
    return singular_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script for Diffusion/Score Model")

    # Experiment name must be provided by the user
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for directory structure.")
    parser.add_argument("--force_first", type=int, default=-1, help="-1 to ignore, manifold dim to specify first sphere coords.")

    # Parse the experiment argument first
    args = parser.parse_args()

    # Load arguments from the JSON file
    args_path = os.path.join("./results", args.experiment, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"No arguments file found at {args_path}")

    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    # Create a new parser to load the rest of the arguments
    parser = argparse.ArgumentParser(description="Evaluation Script for Diffusion/Score Model")
    
    # Add experiment argument again to include it in the final args
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for directory structure.")
    parser.add_argument("--force_first", type=int, default=-1, help="-1 to ignore, manifold dim to specify first sphere coords.")

    for key, value in args_dict.items():
        # Skip the 'experiment' key as it's already added
        if key == 'experiment' or key == "force_first":
            continue
        parser.add_argument(f"--{key}", type=type(value), default=value)

    # Parse all arguments
    args = parser.parse_args()

    evaluate(args)
