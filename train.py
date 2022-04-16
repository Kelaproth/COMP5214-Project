import os
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler 
from torch.nn.utils import clip_grad
# from eval import eval #import evaluation functions
from model.srgan import Generator, Discriminator, TruncatedVGG19
from model.srresnet import SRResNet
from dataset import SRDataset

MODEL_LIST = ['srgan', 'srresnet', 'vit']

basic_configs = {
    # Data
    'data_dir': 'train2014',
    'crop_size': 96,
    'scaling_factor': 4,
    'num_workers': 4,
    'model_name': 'srresnet',

    # Training
    'device': 'cuda', # cpu or cuda
    'batch_size': 16, 
    'checkpoint': None,
    'epochs':100, # number of training iterations
    'start_epoch': 0,
    'eval_per_epochs': 5,
    'learning_rate': 1e-4,
    'lr_scheduler': None,
    'lr_scheduler_parameters': {
        'step_size': 30,
        'gamma': 0.95, 
    },
    'grad_clip': None, # value of grad_clip. default 1.0
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

# Set cuda device here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(basic_configs, model_configs):

    device = torch.device(basic_configs['device'])

    assert basic_configs['model_name'] in MODEL_LIST, "Uknown model."

    if basic_configs['model_name'] == 'srgan':
        if basic_configs['checkpoint']:
            checkpoint = torch.load(basic_configs['checkpoint'])
            generator = checkpoint['generator']
            discriminator = checkpoint['discriminator']
            optimizer_g = checkpoint['optimizer_g']
            optimizer_d = checkpoint['optimizer_d']
            print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))
        else:
            generator = Generator(large_kernel_size=model_configs['large_kernel_size_g'],
                                    small_kernel_size=model_configs['small_kernel_size_g'],
                                    n_channels=model_configs['n_channels_g'],
                                    n_blocks=model_configs['n_blocks_g'],
                                    scaling_factor=basic_configs['scaling_factor'])

            # Initialize generator network with pretrained SRResNet
            generator.initialize_with_srresnet(srresnet_checkpoint=model_configs['srresnet_checkpoint'])

            # Initialize generator's optimizer
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                            lr=basic_configs['learning_rate'])

            # Discriminator
            discriminator = Discriminator(kernel_size=model_configs['kernel_size_d'],
                                            n_channels=model_configs['n_channels_d'],
                                            n_blocks=model_configs['n_blocks_d'],
                                            fc_size=model_configs['fc_size_d'])

            # Initialize discriminator's optimizer
            optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                            lr=basic_configs['learning_rate'])


        # Truncated VGG19 network to be used in the loss calculation
        truncated_vgg19 = TruncatedVGG19(i=model_configs['vgg19_i'], j=model_configs['vgg19_j'])
        truncated_vgg19.eval()

        # Loss functions
        content_loss_criterion = nn.MSELoss()
        adversarial_loss_criterion = nn.BCEWithLogitsLoss()

        generator = generator.to(device)
        discriminator = discriminator.to(device)
        truncated_vgg19 = truncated_vgg19.to(device)

    elif basic_configs['model_name'] == 'srresnet':
        if basic_configs['checkpoint']:
            checkpoint = torch.load(basic_configs['checkpoint'])
            # start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
        else:
            model = SRResNet(large_kernel_size=model_configs['large_kernel_size'],
                                    small_kernel_size=model_configs['small_kernel_size'],
                                    n_channels=model_configs['n_channels'],
                                    n_blocks=model_configs['n_blocks'],
                                    scaling_factor=basic_configs['scaling_factor'])
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=basic_configs['learning_rate'])

        model = model.to(device)
        criterion = nn.MSELoss()

    elif basic_configs['model_name'] == 'vit':
        raise NotImplementedError

        # Custom dataloaders
    train_dataset = SRDataset(basic_configs['data_dir'],
                            type='train',
                            crop_size=basic_configs['crop_size'],
                            scaling_factor=basic_configs['scaling_factor'],
                            lr_img_type='imagenet-norm',
                            hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=basic_configs['batch_size'], 
                            shuffle=True, num_workers=basic_configs['num_workers'],
                                            pin_memory=True)

    # Create learning rate scheduler
    # FIXME: Not finished
    if basic_configs['lr_scheduler']:
        assert basic_configs['lr_scheduler_parameters'] is not None
        if basic_configs['lr_scheduler'] == 'steplr':
            learning_rate_scheduler = lr_scheduler.StepLR(optimizer=optimizer, **basic_configs['lr_scheduler_parameters'])
        else:
            raise NotImplementedError
    else:
        learning_rate_scheduler = None

    if basic_configs['model_name'] in ['srresnet']:
        train_single_model(train_data_loader=train_loader,
                            model=model, 
                            optimizer=optimizer,
                            criterion=criterion,
                            epochs=basic_configs['epochs'],
                            eval_per_epochs=basic_configs['eval_per_epochs'],
                            device=device, 
                            grad_clip=basic_configs['grad_clip'],
                            lr_scheduler=learning_rate_scheduler,
                            save=basic_configs['save'])
    elif basic_configs['model_name'] in ['srgan']:
        train_generative_adversarial_model(train_data_loader=train_loader,
                                            generator=generator,
                                            discriminator=discriminator,
                                            loss_computer=truncated_vgg19,
                                            content_loss_criterion=content_loss_criterion,
                                            adversarial_loss_criterion=adversarial_loss_criterion,
                                            optimizer_g=optimizer_g,
                                            optimizer_d=optimizer_d,
                                            epochs=basic_configs['epochs'],
                                            device=device,
                                            grad_clip=basic_configs['grad_clip'],
                                            lr_scheduler=learning_rate_scheduler)



def train_single_model(train_data_loader, 
                        model, optimizer, criterion, 
                        epochs, eval_per_epochs, device, 
                        grad_clip=None, lr_scheduler=None, save=False):

    model.train()
    
    for epoch in range(epochs):

        start_time = time.time()
        loss_history = []
        with tqdm(train_data_loader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad()

                # Prepare Data
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device) 

                # Forward the model
                sr_imgs_pred = model(lr_imgs)

                loss = criterion(sr_imgs_pred, hr_imgs)
                loss.backward()
                loss_history.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                        max_norm=grad_clip, norm_type=2.0, error_if_nonfinite=False)
                
                if lr_scheduler:
                    lr_scheduler.step()
                
                optimizer.step()

        end_time = time.time()

        loss_val = np.mean(loss_history)
        print('Epoch: {0}----Loss {loss:.4f}'
            '----Time: {time} secs'.format(epoch, 
            loss=loss_val, time=end_time-start_time))
        
        # Do evaluation
        if (epoch + 1) % eval_per_epochs == 0:
            pass

        if save:
            torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   'checkpoint_srresnet.pth.tar')


def train_generative_adversarial_model(train_data_loader, generator, discriminator, 
        loss_computer, content_loss_criterion, adversarial_loss_criterion,
        optimizer_g, optimizer_d, 
        epochs, device, 
        grad_clip=True, lr_scheduler=None):

    pass

if __name__ == '__main__':
    # Set proper network setting
    if basic_configs['model_name'] == 'srresnet':
        model_configs = srresnet_configs
    elif basic_configs['model_name'] == 'srgan':
        model_configs = srgan_configs
    else:
        raise NotImplementedError

    run(basic_configs, model_configs)