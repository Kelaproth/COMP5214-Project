import os
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import ge, optim
from torch import nn
from torch.optim import lr_scheduler 
from torch.nn.utils import clip_grad
from torch.utils.data import random_split
# from eval import eval #import evaluation functions
from model.srgan import Generator, Discriminator, TruncatedVGG19
from model.srresnet import SRResNet
from dataset import SRDataset
from utils import image_converter
import skimage
import lpips

def show_sample_image():
    pass

def test(test_data_loader, model, device):
    
    model.eval()
    psnr, ssim, mse, nmi, dist_= [], [], [], [], []
    loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
    loss_fn_alex.to(device)
    with torch.no_grad():
        with tqdm(test_data_loader, unit='batch') as vepoch:
            for batch in vepoch:

                # Prepare Data
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device) 

                # Forward the model
                sr_imgs_pred = model(lr_imgs) # Will be [-1, 1]
                    
                dist = loss_fn_alex.forward(sr_imgs_pred, hr_imgs)
                dist_.append(dist.mean().item())
                
                # yimgs = image_converter(sr_imgs_pred.cpu().detach().numpy(), 
                #     source='[-1, 1]', target='[0, 255]')
                # gtimgs = image_converter(hr_imgs.cpu().detach().numpy(), 
                #     source='[-1, 1]', target='[0, 255]')
                yimgs = sr_imgs_pred.cpu().detach().numpy() # Simple [-1, 1]
                gtimgs = hr_imgs.cpu().detach().numpy()
                for yimg, gtimg in zip(yimgs, gtimgs):
                    psnr.append(skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
                    ssim.append(skimage.metrics.structural_similarity(yimg, gtimg, channel_axis=0))
                    mse.append(skimage.metrics.mean_squared_error(yimg, gtimg))
                    nmi.append(skimage.metrics.normalized_mutual_information(yimg, gtimg))

    print("Test phase")
    print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
        % (np.mean(psnr), np.mean(ssim), 
            np.mean(mse), np.mean(nmi), np.mean(dist_)))

def test_on_multiple_dataset(datasets, model_path, device):

    if 'gan' in model_path:
        model = torch.load(model_path)['generator'].to(device)
    else:
        model = torch.load(model_path)['model'].to(device)

    for dataset in datasets:
        test_dataset = SRDataset(dataset,
                        type='test',
                        crop_size=0,
                        scaling_factor=4,
                        lr_img_type='imagenet-norm',
                        hr_img_type='[-1, 1]')
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                            batch_size=16, 
                            shuffle=True, num_workers=4,
                                            pin_memory=True)                           
        test(test_loader, model, device)