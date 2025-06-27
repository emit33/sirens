import cv2
import torch


def load_and_preprocess_img(img_path):
    img = cv2.imread(img_path)
    img = (img - img.min()) / (img.max() - img.min())  # Normalise
    img = img.transpose(2, 1, 0)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)  # BxCxHxW
    return img
