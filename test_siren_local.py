import torch
from torch import nn
from utils.SirenNet import SirenNet, SirenWrapper
import cv2
import matplotlib.pyplot as plt

from utils.preprocessing import load_and_preprocess_img
from utils.train import train_net


if __name__ == "__main__":
    # Parameter set up
    img_path = "/home/tempus/projects/data/triangles/triangle1.png"
    n_epochs = 100
    lr = 0.01

    # ------------------ #
    # Additional (non-configurable) parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ #
    # Run script
    # Load and preprocess image
    img = load_and_preprocess_img(img_path, 256)
    img = img.to("cuda")

    reconstructed_img, net, losses = train_net(
        img,
        dim_in=2,
        dim_hidden=256,
        dim_out=1,
        n_hidden_layers=5,
        final_activation=nn.Sigmoid(),
        w0=30.0,
        device=torch.device("cuda"),
        n_epochs=100,
        lr=1e-2,
    )

    plt.imshow(reconstructed_img, cmap="gray")
    plt.savefig("siren_output_local.png")
