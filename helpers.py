from __future__ import annotations
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import math
import matplotlib.pyplot as plt
import urllib
import numpy as np
import PIL
import subprocess
import os
from typing import List, Tuple, Dict, Any, Optional, Callable
import csv
import pathlib
import time as ptime

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

IMAGE_SHAPE = (32, 32)

img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SHAPE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (2 * x) - 1)
])


img_transform_inverse = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1) / 2),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)), # 0th dim, i.e., channels is put at the end, no batch size here.
    transforms.Lambda(lambda x: x * 255.),
    transforms.Lambda(lambda x: x.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])


class AbstractDiffusionModel:
    @staticmethod
    def gather_from_list(values: torch.Tensor, indices: torch.Tensor, ndims: int) -> torch.Tensor:
        """
        Gathers elements from a one-dimensional tensor at specific indices and reshapes the
        resulting tensor to have `ndims` dimensions (plus a batch-size dimension).
        
        Args:
            values: Tensor of shape (N,).
            indices: Tensor of shape (B,).
            ndims: Number of dimensions the final tensor has (plus batch dimension).
            
        Returns:
            Tensor of dimensions ndims + 1 of shape (B, 1, 1, ...) with elements from values.
        """
        batch_size = indices.shape[0]
        val_at_idx = values.gather(-1, indices).reshape(batch_size, *[1 for _ in range(ndims)])
        return val_at_idx
    

def get_shiba() -> PIL.Image.Image:
    filename = 'shiba.jpg'
    if not os.path.exists(filename):
        # url = 'https://www.shutterstock.com/image-vector/cute-shiba-inu-dog-paws-260nw-1429960835.jpg'
        url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.istockphoto.com%2Fvectors%2Fjapanese-dog-shiba-inu-logo-design-template-vector-id1256108875%3Fk%3D20%26m%3D1256108875%26s%3D170667a%26w%3D0%26h%3DZpLZGkh6qjXWbGS3T1BTzBEbHAiAia8xXl26_2VKd_E%3D&f=1&nofb=1&ipt=45e40780a2f1bb91a9d75bdd2e6388978fe6ee49c03a451a730f4cf9cb35ab43&ipo=images"
        urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)


def get_streetsigns() -> PIL.Image.Image:
    filename = "streetsigns.png"
    if not os.path.exists(filename):
        url = "https://repository-images.githubusercontent.com/450090314/d4de1683-868d-49b9-ab47-d371252b2b99"
        urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)


def visualize_forward_diffusion(img: PIL.Image.Image, diffusion_model: AbstractDiffusionModel,
                               num_images_display=5, device='cpu'):
    torch_image_batch = torch.stack([img] * num_images_display)
    t = torch.linspace(0, diffusion_model.num_timesteps - 1, num_images_display).long()
    noisy_image_batch, _ = diffusion_model.forward(torch_image_batch, t, device)
    
    f, ax = plt.subplots(1, num_images_display + 1, figsize=(2 * (num_images_display + 1), 2))
    ax[0].imshow(img_transform_inverse(img))
    ax[0].axis('off')
    ax[0].set_title("Step: 0", fontsize=12)
    for idx, image in enumerate(noisy_image_batch):
        ax[idx + 1].imshow(img_transform_inverse(image))
        ax[idx + 1].axis('off')
        ax[idx + 1].set_title(f"Step: {t[idx].item()}", fontsize=12)
    plt.show()
    plt.close("all")
    
    
def plot_encoding(encoding: PositionalEncoding):
    ax = plt.subplot()
    cax = ax.imshow(encoding(torch.Tensor(np.arange(128))).detach().numpy())
    ax.axis('off')
    ax.set_title("Positional encoding")
    plt.show()
    plt.close("all")
    
    
def plot_noise_prediction(noise, predicted_noise):
    plt.figure(figsize=(15,15))
    f, ax = plt.subplots(1, 2, figsize=(5,5))
    ax[0].imshow(img_transform_inverse(noise))
    ax[0].set_title(f"ground truth noise", fontsize=10)
    ax[1].axis('off')
    ax[1].imshow(img_transform_inverse(predicted_noise))
    ax[1].set_title(f"predicted noise", fontsize=10)
    ax[1].axis('off')
    plt.show()
    
    
def plot_noise_distribution(noise, predicted_noise):
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(6)
    fig.set_figheight(6 * 0.61)
    ax.hist(noise.cpu().numpy().flatten(), density=True, alpha=0.8, bins=30, label="ground truth noise")
    ax.hist(predicted_noise.cpu().numpy().flatten(), density=True, alpha=0.8, bins=30, label="predicted noise")
    ax.legend(frameon=False)
    ax.set_xlabel("Noise values")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    
def visualize_image(img: PIL.Image.Image, t: int | None = None):
    plt.figure(figsize=(2,2))
    plt.imshow(img_transform_inverse(img[0]))
    if t is not None:
        plt.title(f"Step {t}")
    plt.axis('off')
    plt.show()
    plt.close('all')
    
    
def visualize_trained_classconditioned_unet(diffusion_model, unet: torch.Module, 
                                            class_names: List[str], class_indices: List[int],
                                            num_imgs_to_display: int,
                                            device: torch.device, image_shape=(32,32)):
    num_classes = len(class_names)
    torch.manual_seed(16)
    f, ax = plt.subplots(num_classes, num_imgs_to_display, figsize=(2 * num_imgs_to_display, 2 * num_classes))

    with torch.no_grad():
        for j, (cidx, name) in enumerate(zip(class_indices, class_names)):
            imgs = torch.randn((num_imgs_to_display, 3) + image_shape).to(device)
            for i in reversed(range(diffusion_model.num_timesteps)):
                t = torch.full((1,), i, dtype=torch.long, device=device)
                labels = torch.nn.functional.one_hot(torch.tensor([cidx] * num_imgs_to_display), unet.num_classes).float().to(device)
                imgs = diffusion_model.backward(x_t=imgs, t=t, model=unet.eval().to(device), labels=labels)
            for idx, img in enumerate(imgs):
                ax[j][idx].imshow(img_transform_inverse(img))
                ax[j][idx].set_title(f"Class: {name}", fontsize=8)
                ax[j][idx].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(f)
    
    
def verify_diffusion_init(diffusion_model):
    actual_betas = torch.linspace(diffusion_model.schedule_start, diffusion_model.schedule_end, diffusion_model.num_timesteps)
    actual_alphas = 1 - diffusion_model.betas
    actual_overline_alphas = torch.cumprod(diffusion_model.alphas, axis=0)
    
    np.testing.assert_almost_equal(
        desired=actual_betas.detach().numpy(),
        actual=diffusion_model.betas.detach().numpy(),
        decimal=5,
        err_msg='',
        verbose=True
    )
    np.testing.assert_almost_equal(
        desired=actual_alphas.detach().numpy(),
        actual=diffusion_model.alphas.detach().numpy(),
        decimal=5,
        err_msg='',
        verbose=True
    )
    np.testing.assert_almost_equal(
        desired=actual_overline_alphas.detach().numpy(),
        actual=diffusion_model.overline_alphas.detach().numpy(),
        decimal=5,
        err_msg='',
        verbose=True
    )
    print("All tests passed, your diffusion model gets correctly initialized!")
    
    
def verify_positional_encoding_init(encoding):
    actual_dim = encoding.dim
    actual_half_dim = actual_dim // 2
    actual_embedding = np.log(10000) / (actual_half_dim - 1)
    actual_embedding = torch.exp(torch.arange(actual_half_dim, dtype=torch.float32) * (-1. * actual_embedding))
    np.testing.assert_almost_equal(
        actual=encoding.omegas,
        desired=actual_embedding.detach().numpy(),
        decimal=5,
        err_msg="The \\omega parameters do not match.",
        verbose=True
    )
    print("All tests passed, your positional encoding gets correctly initialized!")
    
    
def verify_positional_encoding_forward(encoding):
    actual_dim = encoding.dim
    actual_half_dim = actual_dim // 2
    actual_embedding = np.log(10000) / (actual_half_dim - 1)
    actual_embedding = torch.exp(torch.arange(actual_half_dim, dtype=torch.float32) * (-1. * actual_embedding))
    
    position = torch.arange(34)
    actual_embedding = position.unsqueeze(1) * actual_embedding.unsqueeze(0)
    actual_embedding = torch.cat([torch.sin(actual_embedding), torch.cos(actual_embedding)], dim=-1)
    np.testing.assert_almost_equal(
        actual=encoding(position).detach().numpy(),
        desired=actual_embedding.detach().numpy(),
        decimal=5,
        err_msg="The computation of the embeddings in the forward pass do not match.",
        verbose=True
    )
    print("All tests passed, your positional encoding computes the correct values!")