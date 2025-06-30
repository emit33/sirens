from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sine(nn.Module):
    def __init__(self, w_0):
        super().__init__()
        self.w_0 = w_0

    def forward(self, x):
        return torch.sin(self.w_0 * x)


class SirenLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w_0=1.0,
        c=6.0,
        is_first=False,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in

        # Set up weight and bias
        weight = torch.zeros((dim_out, dim_in))
        bias = torch.zeros(dim_out)
        self.init_(weight, bias, c, w_0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        # Set up activation
        self.activation = activation if activation else Sine(w_0)

    def init_(self, weight, bias, c, w_0):
        """
        Warning: differs from implementation
        """
        max = np.sqrt(c / self.dim_in)

        weight.uniform_(-max, max)
        bias.uniform_(-max, max)

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_hidden_layers,
        w_0=30.0,
        w_0_initial=1.0,
        c=6.0,
        final_activation=None,
    ):
        super().__init__()
        self.num_hidden_layers = n_hidden_layers
        self.dim_hidden = dim_hidden

        self.layers = self._build_layers(
            dim_in,
            dim_hidden,
            dim_out,
            w_0,
            w_0_initial,
            n_hidden_layers,
            final_activation,
        )

    def _build_layers(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        w_0,
        w_0_initial,
        n_hidden_layers,
        final_activation,
    ):
        # Obtain all layers except last

        layers = nn.ModuleList([])
        for i in range(n_hidden_layers):
            # Set paramters that are different if we work with the first lyaer
            is_first = i == 0
            layer_dim_in = dim_in if is_first else dim_hidden
            layer_w_0 = w_0 if is_first else 1

            layer = SirenLayer(layer_dim_in, dim_hidden, layer_w_0, is_first=is_first)
            layers.append(layer)

        # Add final layer
        final_activation = (
            nn.Identity() if final_activation is None else final_activation
        )
        last_layer = SirenLayer(
            dim_hidden, dim_out, w_0=w_0, is_first=False, activation=final_activation
        )

        layers.append(last_layer)

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class SirenWrapper(nn.Module):
    def __init__(self, siren_net, image_width, image_height, grayscale_flag):
        super().__init__()
        self.siren_net = siren_net
        self.image_width = image_width
        self.image_height = image_height
        self.grayscale_flag = grayscale_flag

        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.meshgrid(tensors, indexing="ij")
        mgrid = torch.stack(mgrid, dim=-1)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)
        self.grid: torch.Tensor

    def forward(self, img=None):
        coords = self.grid.clone().detach().requires_grad_()
        out = self.siren_net(coords)

        if self.grayscale_flag:
            out = out.squeeze()  # remove singleton channel
            out = rearrange(
                out, "(h w) -> () h w", h=self.image_height, w=self.image_width
            )
        else:
            out = rearrange(
                out, "(h w) c -> () c h w", h=self.image_height, w=self.image_width
            )

        if img is not None:
            loss = F.mse_loss(out, img)
            return loss

        return out
