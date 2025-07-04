import torch
import torch.nn as nn
from utils.SirenNet import SirenNet, SirenWrapper


def train_net(
    img,
    dim_in=2,
    dim_hidden=256,
    dim_out=1,
    n_hidden_layers=5,
    w0=30.0,
    n_epochs=100,
    lr=1e-2,
    final_activation=nn.Sigmoid(),
    grayscale_flag=True,
    device=torch.device("cuda"),
):
    net = SirenNet(
        dim_in=dim_in,  # input dimension, ex. 2d coor
        dim_hidden=dim_hidden,  # hidden dimension
        dim_out=dim_out,  # output dimension, ex. rgb value
        n_hidden_layers=n_hidden_layers,  # number of layers
        final_activation=final_activation,  # activation of final layer (nn.Identity() for direct output)
        w_0_initial=w0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
    )
    net.to(device)

    wrapper = SirenWrapper(
        net,
        image_width=img.shape[-1],
        image_height=img.shape[-1],
        grayscale_flag=grayscale_flag,
    )
    wrapper.to(device)

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr)

    # Train
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = wrapper(img)

        print(f"Epoch: {epoch}; Loss: {loss}")
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    reconstructed_img = wrapper()

    if grayscale_flag:
        # [B,H,W] -> [B,C,H,W] add color channel
        reconstructed_img = reconstructed_img.unsqueeze(1).repeat(1, 3, 1, 1)

    # [B,C,H,W] -> [H,W,C]
    reconstructed_img = reconstructed_img[0].permute(1, 2, 0).cpu().detach().numpy()

    return reconstructed_img, net, losses
