import torch
from torch import nn
from utils.SirenNet import SirenNet, SirenWrapper
import cv2
import matplotlib.pyplot as plt

# Parameter set up
n_epochs = 100
lr = 0.01

# ------------------ #
# Additional (non-configurable) parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
img = cv2.imread("images/triangle1.png")
img = img[:, :, 0:1]  # Make grayscale; keep final dim
img = (img - img.min()) / (img.max() - img.min())  # Normalise
img = img.transpose(2, 1, 0)
img = torch.from_numpy(img).float()
img = img.unsqueeze(0)  # BxCxHxW
img = img.to(device)

# Set up SIREN
net = SirenNet(
    dim_in=2,  # input dimension, ex. 2d coor
    dim_hidden=256,  # hidden dimension
    dim_out=1,  # output dimension, ex. rgb value
    n_hidden_layers=5,  # number of layers
    final_activation=nn.Sigmoid(),  # activation of final layer (nn.Identity() for direct output)
    w_0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
)
net.to(device)

wrapper = SirenWrapper(net, image_width=480, image_height=640)
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


# Infer
pred_img = wrapper()  # (1, 3, 256, 256)

img_np = pred_img[0].permute(2, 1, 0).cpu().detach().numpy()
plt.imshow(img_np, cmap="gray")
plt.savefig("siren_output_local.png")

pass
