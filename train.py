import os
import time
from tqdm import tqdm
import torch
from torch import optim
from torch.optim import lr_scheduler 
from torch.nn.utils import clip_grad
from eval import eval #import evaluation functions
basic_configs = {
    # Data
    'data_dir': './data',
    'crop_size': 96,
    'scaling_factor': 4,
    'num_workers': 4,

    # Training
    'device': 'cuda', # cpu or cuda
    'batch_size': 16, 
    'epochs':100, # number of training iterations
    'eval_per_epochs': 5,
    'learning_rate': 1e-4,
    'lr_scheduler': True,
    'lr_scheduler_parameters': {
        'scheduler_name': None,
        'gamma': 0.95, 
    },
    'grad_clip': False,
    'save': True,
}

model_configs = {
    
}

# Set device here
device = torch.device(basic_configs['device'])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(configs):

    
    
    
    pass


def train(train_data_loader, model, optimizer, device=device, 
        grad_clip=True, lr_scheduler=True):

    model.train()

    start_time = time.time()
    
    for i, batch in tqdm(enumerate(train_data_loader)):
        optimizer.zero_grad()

        lr_imgs, hr_imgs = batch
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device) 

        sr_imgs_pred = model(lr_imgs)

        loss = model.loss(sr_imgs_pred, hr_imgs)

        loss.backward()

        if grad_clip:
            pass

        optimizer.step()


