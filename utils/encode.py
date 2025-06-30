import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from utils.config import ModelConfig, PathConfig, TrainingConfig
from utils.preprocessing import load_and_preprocess_img, preprocess_tensor
from utils.train import train_net


def encode_dir(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    paths_config: PathConfig,
    final_activation=nn.Sigmoid(),
    device=torch.device("cuda"),
):
    img_paths = os.listdir(paths_config.data_dir)
    img_paths = [Path(paths_config.data_dir) / img_path for img_path in img_paths]

    # Remove old results directory if it exists
    if os.path.exists(paths_config.results_dir):
        shutil.rmtree(paths_config.results_dir)

    # Create one directory for reconstructed images and one for networks
    reconstructed_imgs_dir = paths_config.results_dir / "reconstructed_imgs"
    os.makedirs(reconstructed_imgs_dir)

    ckpts_dir = paths_config.results_dir / "ckpts"
    os.makedirs(ckpts_dir)

    for img_path in img_paths:
        img = load_and_preprocess_img(
            img_path, training_config.resolution, training_config.grayscale
        )
        img = img.to(device)

        reconstructed_img, net, losses = train_net(
            img,
            grayscale_flag=training_config.grayscale,
            dim_in=model_config.dim_in,
            dim_hidden=model_config.dim_hidden,
            dim_out=model_config.dim_out,
            n_hidden_layers=model_config.n_hidden_layers,
            w0=model_config.w0,
            n_epochs=training_config.n_epochs,
            lr=training_config.lr,
            final_activation=final_activation,
            device=device,
        )

        # Save reconstructed image
        plt.imshow(reconstructed_img, cmap="gray")
        plt.savefig(reconstructed_imgs_dir / img_path.name)

        # Save model
        torch.save(net.state_dict(), ckpts_dir / f"{img_path.stem}.pt")


def encode_data_tensor(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    paths_config: PathConfig,
    final_activation=nn.Sigmoid(),
    device=torch.device("cuda"),
):
    # Load images tensor
    data_tensor = torch.load(paths_config.data_dir / "images.pt")

    # Remove old results directory if it exists, and create a directory to save the checkpoints in
    if os.path.exists(paths_config.results_dir):
        shutil.rmtree(paths_config.results_dir)

    ckpts_dir = paths_config.results_dir / "ckpts"
    os.makedirs(ckpts_dir)

    for i, img in enumerate(data_tensor):
        img = preprocess_tensor(img)
        img = img.to(device)

        reconstructed_img, net, losses = train_net(
            img,
            dim_in=model_config.dim_in,
            dim_hidden=model_config.dim_hidden,
            dim_out=model_config.dim_out,
            n_hidden_layers=model_config.n_hidden_layers,
            w0=model_config.w0,
            n_epochs=training_config.n_epochs,
            lr=training_config.lr,
            final_activation=final_activation,
            device=device,
        )

        # Save model
        torch.save(net.state_dict(), ckpts_dir / f"{i}.pt")
