import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T

from utils import preprocess, recover_image

import argparse


to_pil = T.ToPILImage()


# make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details

model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
# model_id_or_path = "CompVis/stable-diffusion-v1-3"
# model_id_or_path = "CompVis/stable-diffusion-v1-2"
# model_id_or_path = "CompVis/stable-diffusion-v1-1"

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)
pipe_img2img = pipe_img2img.to("cuda")

def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])

        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

    return X_adv


def main():
    
    img = args.input_dir

    init_image = Image.open(img).convert("RGB")
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_image = center_crop(resize(init_image))

    with torch.autocast('cuda'):
        X = preprocess(init_image).half().cuda()
        adv_X = pgd(X,
                    model=pipe_img2img.vae.encode,
                    clamp_min=-1,
                    clamp_max=1,
                    eps=0.06, # The higher, the less imperceptible the attack is
                    step_size=0.02, # Set smaller than eps
                    iters=500, # The higher, the stronger your attack will be
                   )
    
        # convert pixels back to [0,1] range
        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    
        adv_image = to_pil(adv_X[0]).convert("RGB")
    adv_image.save(args.output_dir, 'png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Input image dir')
    parser.add_argument('--output_dir', required=True,
                        help='Output image dir')

    args = parser.parse_args()

    main()
    
