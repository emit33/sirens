import cv2
import torch


def load_and_preprocess_img(img_path, res, grayscale_flag=True):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    img = (img - img.min()) / (img.max() - img.min())  # Normalise
    if grayscale_flag:
        img = img[:, :, 0]  # [H,W,C] -> [H,W]
    else:
        img = img.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W,]
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)  # [B,C,H,W] or [B,H,W]
    return img


def preprocess_tensor(img):
    img = (img - img.min()) / (img.max() - img.min())  # Normalise
    img = img.unsqueeze(0)  # [B,C,H,W] or [B,H,W]

    return img
