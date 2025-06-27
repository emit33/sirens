import torch
import torch.nn as nn
from utils.SirenNet import SirenNet, SirenWrapper


def train_net(
    img,
    dim_in=2,
    dim_hidden=256,
    dim_out=1,
    n_hidden_layers=5,
    final_activation=nn.Sigmoid(),
    w_0_initial=30.0,
    device=torch.device("cuda"),
    n_epochs=100,
    lr=1e-2,
):
    net = SirenNet(
        dim_in=dim_in,  # input dimension, ex. 2d coor
        dim_hidden=dim_hidden,  # hidden dimension
        dim_out=dim_out,  # output dimension, ex. rgb value
        n_hidden_layers=n_hidden_layers,  # number of layers
        final_activation=final_activation,  # activation of final layer (nn.Identity() for direct output)
        w_0_initial=w_0_initial,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
    )
    net.to(device)

    wrapper = SirenWrapper(net, image_width=256, image_height=256)
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

    reconstructed_img = reconstructed_img[0].permute(2, 1, 0).cpu().detach().numpy()

    return reconstructed_img, net, losses
