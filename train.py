import os
import time
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler 
from torch.nn.utils import clip_grad
from eval import eval #import evaluation functions
from model.srgan import Generator, Discriminator, TruncatedVGG19
from model.srresnet import SRResNet
from dataset import SRDataset

MODEL_LIST = ['srgan', 'srresnet', 'vit']

basic_configs = {
    # Data
    'data_dir': './data',
    'crop_size': 96,
    'scaling_factor': 4,
    'num_workers': 4,

    # Training
    'device': 'cuda', # cpu or cuda
    'batch_size': 16, 
    'checkpoint': None,
    'epochs':100, # number of training iterations
    'start_epoch': 0,
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

srresnet_configs = {
    'large_kernel_size': 9,
    'small_kernel_size': 3,
    'n_channels': 64,
    'n_blocks': 16,
}

srgan_configs = {
    'large_kernel_size_g': 9,
    'small_kernel_size_g': 3,
    'n_channels_g': 64,
    'n_blocks_g': 16,

    'srresnet_checkpoint': "./checkpoint_srresnet.pth.tar",
    'kernel_size_d': 3,
    'n_channels_d': 64,
    'n_blocks_d': 8,
    'fc_size_d': 1024,
    'vgg19_i': 5,
    'vgg19_j': 4,
    'beta': 1e-3,
}

# Set device here
device = torch.device(basic_configs['device'])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(basic_configs, model_name, model_configs):

    assert model_name in MODEL_LIST, "Uknown model."

    if model_name == 'srresnet':
        if basic_configs['checkpoint']:
            checkpoint = torch.load(basic_configs['checkpoint'])
            # start_epoch = checkpoint['epoch'] + 1
            generator = checkpoint['generator']
            discriminator = checkpoint['discriminator']
            optimizer_g = checkpoint['optimizer_g']
            optimizer_d = checkpoint['optimizer_d']
            print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))
        else:
            # Generator
            generator = Generator(large_kernel_size=srgan_configs['large_kernel_size_g'],
                                    small_kernel_size=srgan_configs['small_kernel_size_g'],
                                    n_channels=srgan_configs['n_channels_g'],
                                    n_blocks=srgan_configs['n_blocks_g'],
                                    scaling_factor=srgan_configs['scaling_factor'])

            # Initialize generator network with pretrained SRResNet
            generator.initialize_with_srresnet(srresnet_checkpoint=srgan_configs['srresnet_checkpoint'])

            # Initialize generator's optimizer
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                            lr=basic_configs['learning_rate'])

            # Discriminator
            discriminator = Discriminator(kernel_size=srgan_configs['kernel_size_d'],
                                            n_channels=srgan_configs['n_channels_d'],
                                            n_blocks=srgan_configs['n_blocks_d'],
                                            fc_size=srgan_configs['fc_size_d'])

            # Initialize discriminator's optimizer
            optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                            lr=basic_configs['learning_rate'])


        # Truncated VGG19 network to be used in the loss calculation
        truncated_vgg19 = TruncatedVGG19(i=srgan_configs['vgg19_i'], j=srgan_configs['vgg19_j'])
        truncated_vgg19.eval()

        # Loss functions
        content_loss_criterion = nn.MSELoss()
        adversarial_loss_criterion = nn.BCEWithLogitsLoss()

        # Move to default device
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        truncated_vgg19 = truncated_vgg19.to(device)
        content_loss_criterion = content_loss_criterion.to(device)
        adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    elif model_name == 'srgan':
        if basic_configs['checkpoint']:
            checkpoint = torch.load(basic_configs['checkpoint'])
            # start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
        else:
            model = SRResNet(large_kernel_size=srgan_configs['large_kernel_size'],
                                    small_kernel_size=srgan_configs['small_kernel_size'],
                                    n_channels=srgan_configs['n_channels'],
                                    n_blocks=srgan_configs['n_blocks'],
                                    scaling_factor=srgan_configs['scaling_factor'])
            # Initialize the optimizer
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=basic_configs['learning_rate'])

        # Move to default device
        model = model.to(device)
        criterion = nn.MSELoss().to(device)
    elif model_name == 'vit':
        raise NotImplementedError

        # Custom dataloaders
    train_dataset = SRDataset(basic_configs['data_dir'],
                            split='train',
                            crop_size=basic_configs['crop_size'],
                            scaling_factor=basic_configs['scaling_factor'],
                            lr_img_type='imagenet-norm',
                            hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=basic_configs['batch_size'], 
                            shuffle=True, num_workers=basic_configs['workers'],
                                            pin_memory=True)



def train_single_model(train_data_loader, model, optimizer, device=device, 
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

def train_generative_adversarial_model(train_loader, generator, discriminator, 
        truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
        optimizer_g, optimizer_d, epoch, device=device, 
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
