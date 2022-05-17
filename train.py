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
from dataset import SRSamplingDataset
from utils import image_converter
import skimage
import lpips
from model.mae import prepare_mae
from model.ipt import args_ipt, Checkpoint, Loss, Model, quantize

MODEL_LIST = ['srgan', 'srresnet', 'vit', 'mae', 'ipt']

basic_configs = {
    # Data
    # 'data_dir': 'div2k',
    'data_dir': 'val2014',
    'crop_size': 96,
    'scaling_factor': 4,
    'num_workers': 4,
    'val_proportion': 0.05,

    # Training
    'model_name': 'mae',
    'device': 'cuda', # cpu or cuda
    'batch_size': 16, 
    'checkpoint': None,
    'epochs':100, # number of training iterations
    # 'start_epoch': 0,
    'eval_per_epochs': 5,
    'learning_rate': 1e-5,
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

    'srresnet_checkpoint': './save/srresnet_checkpoint_99.pth.tar',
    'kernel_size_d': 3,
    'n_channels_d': 64,
    'n_blocks_d': 8,
    'fc_size_d': 1024,
    'vgg19_i': 5,
    'vgg19_j': 4,
    'beta': 1e-3,
}

ipt_configs = args_ipt

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

    elif basic_configs['model_name'] == 'mae':
        model, optimizer = prepare_mae()
        criterion = nn.MSELoss()
    elif basic_configs['model_name'] == 'ipt':
        # Manually input rather than process it here
        # model_configs['scale'] = list(map(lambda x: int(x), model_configs['scale'].split('+')))
        # model_configs['data_train'] =  model_configs['data_train'].split('+')
        # model_configs['data_test'] = model_configs['data_test'].split('+')
        checkpoint = Checkpoint(model_configs)
        if checkpoint.ok:
            model = Model(model_configs, checkpoint)
            if basic_configs['checkpoint']:
                if model_configs['pretrain'] == '':
                    model_configs['pretrain'] = "./save/ipt/IPT_sr4.pt"
                state_dict = torch.load(model_configs['pretrain'])
                model.model.load_state_dict(state_dict, strict = False)

                if model_configs['optimizer'] == 'ADAM':
                    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                            lr=basic_configs['learning_rate'], betas=model_configs['betas'],
                            eps=model_configs['epsilon'], weight_decay=model_configs['weight_decay'])
                elif model_configs['optimizer'] == 'SGD':
                    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                            lr=basic_configs['learning_rate'], momentum=model_configs['momentum'],
                            weight_decay=model_configs['weight_decay'])
                elif model_configs['optimizer'] == 'RMSprop':
                    optimizer = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
                            lr=basic_configs['learning_rate'], 
                            eps=model_configs['epsilon'], weight_decay=model_configs['weight_decay'],
                            momentum=model_configs['momentum'])
                model = model.to(device)
                criterion = Loss(model_configs, ckp=checkpoint)
        else:
            raise Exception

    if basic_configs['model_name'] == 'ipt':
                # Custom dataloaders. Note ipt have intrinsic shiftmean.
        full_train_dataset = SRSamplingDataset(basic_configs['data_dir'],
                                type='train',
                                crop_size=basic_configs['crop_size'],
                                scaling_factor=basic_configs['scaling_factor'],
                                lr_img_type='[0, 255]',
                                hr_img_type='[0, 255]')
    else:
        # Custom dataloaders. Note hr is [-1, 1] and lr is normed for training.
        full_train_dataset = SRSamplingDataset(basic_configs['data_dir'],
                                type='train',
                                crop_size=basic_configs['crop_size'],
                                scaling_factor=basic_configs['scaling_factor'],
                                lr_img_type='imagenet-norm',
                                hr_img_type='[-1, 1]')

    val_size = int(basic_configs['val_proportion'] * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    # Train/Valid split
    train_dataset, val_dataset = \
        random_split(full_train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=basic_configs['batch_size'], 
                            shuffle=True, num_workers=basic_configs['num_workers'],
                                            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=basic_configs['batch_size'], 
                            shuffle=True, num_workers=basic_configs['num_workers'],
                                            pin_memory=True)                                        

    # Create learning rate scheduler
    # FIXME: Not completed
    if basic_configs['lr_scheduler']:
        assert basic_configs['lr_scheduler_parameters'] is not None
        if basic_configs['lr_scheduler'] == 'steplr':
            learning_rate_scheduler = lr_scheduler.StepLR(optimizer=optimizer, **basic_configs['lr_scheduler_parameters'])
        else:
            raise NotImplementedError
    else:
        learning_rate_scheduler = None

    # Start Training
    if basic_configs['model_name'] in ['srresnet']:
        train_single_model(model_name=basic_configs['model_name'],
                            train_data_loader=train_loader,
                            valid_data_loader=val_loader,
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
        train_generative_adversarial_model(model_name=basic_configs['model_name'],
                                            train_data_loader=train_loader,
                                            valid_data_loader=val_loader,
                                            generator=generator,
                                            discriminator=discriminator,
                                            loss_computer=truncated_vgg19,
                                            content_loss_criterion=content_loss_criterion,
                                            adversarial_loss_criterion=adversarial_loss_criterion,
                                            optimizer_generator=optimizer_g,
                                            optimizer_discriminator=optimizer_d,
                                            epochs=basic_configs['epochs'],
                                            beta=model_configs['beta'],
                                            eval_per_epochs=basic_configs['eval_per_epochs'],
                                            device=device,
                                            grad_clip=basic_configs['grad_clip'],
                                            lr_scheduler=learning_rate_scheduler,
                                            save=basic_configs['save'])
    elif basic_configs['model_name'] in ['mae']:
        train_mae(model_name=basic_configs['model_name'],
                                            train_data_loader=train_loader,
                                            valid_data_loader=val_loader,
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            epochs=basic_configs['epochs'],
                                            eval_per_epochs=basic_configs['eval_per_epochs'],
                                            device=device,
                                            grad_clip=basic_configs['grad_clip'],
                                            lr_scheduler=learning_rate_scheduler,
                                            save=basic_configs['save']
                                            )
    elif basic_configs['model_name'] in ['ipt']:
        train_ipt(model_name=basic_configs['model_name'],
                                            train_data_loader=train_loader,
                                            valid_data_loader=val_loader,
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            epochs=basic_configs['epochs'],
                                            eval_per_epochs=basic_configs['eval_per_epochs'],
                                            device=device,
                                            grad_clip=basic_configs['grad_clip'],
                                            lr_scheduler=learning_rate_scheduler,
                                            save=basic_configs['save'])



########################################################
### Training Mae. Model is initilaized in model.mae  ###
########################################################
def train_mae(model_name, train_data_loader, valid_data_loader, model, optimizer, criterion, 
                        epochs, eval_per_epochs, device, 
                        grad_clip=None, lr_scheduler=None, save=False):

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        loss_history = []
        with tqdm(train_data_loader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                optimizer.zero_grad()

                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                # print(lr_imgs.size(), hr_imgs.size(), hr_imgs.size()[2])
                _, _, sr_imgs_pred = model(lr_imgs)

                loss = criterion(sr_imgs_pred, hr_imgs)
                loss.backward()
                loss_history.append(loss.item())
                tepoch.set_postfix(loss=loss.item())


        end_time = time.time()
        loss_train = np.mean(loss_history)
        
        print('Epoch: {0}----Loss {loss:.4f}'
            '----Time: {time} secs'.format(epoch, 
            loss=loss_train, time=end_time-start_time))

        # Do evaluation
        if (epoch + 1) % eval_per_epochs == 0 or epoch==0:
            start_time = time.time()
            model.eval()
            loss_history = []
            psnr, ssim, mse, nmi, dist_= [], [], [], [], []
            loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
            loss_fn_alex.to(device)
            with torch.no_grad():
                with tqdm(valid_data_loader, unit='batch') as vepoch:
                    for batch in vepoch:
                        vepoch.set_description(f"Valid at epoch {epoch}")

                        # Prepare Data
                        lr_imgs, hr_imgs = batch
                        lr_imgs = lr_imgs.to(device)
                        hr_imgs = hr_imgs.to(device) 

                        # Forward the model
                        _, _, sr_imgs_pred = model(lr_imgs) # Will be [-1, 1]

                        loss = criterion(sr_imgs_pred, hr_imgs)
                        loss_history.append(loss.item())
                        vepoch.set_postfix(loss=loss.item())
                            
                        dist = loss_fn_alex.forward(sr_imgs_pred, hr_imgs)
                        dist_.append(dist.mean().item())
                        
                        # yimgs = image_converter(sr_imgs_pred.cpu().detach().numpy(), 
                        #     source='[-1, 1]', target='[0, 255]')
                        # gtimgs = image_converter(hr_imgs.cpu().detach().numpy(), 
                        #     source='[-1, 1]', target='[0, 255]')
                        yimgs = sr_imgs_pred.cpu().detach().numpy() # Simple [-1, 1]
                        gtimgs = hr_imgs.cpu().detach().numpy()
                        for yimg, gtimg in zip(yimgs, gtimgs):
                            psnr.append(skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=2))
                            ssim.append(skimage.metrics.structural_similarity(yimg, gtimg, channel_axis=0))
                            mse.append(skimage.metrics.mean_squared_error(yimg, gtimg))
                            nmi.append(skimage.metrics.normalized_mutual_information(yimg, gtimg))
                        
            end_time = time.time()
            print('Valid at epoch: {0}----Loss {loss:.4f}'
                '----Time: {time} secs'.format(epoch, 
                loss=np.mean(loss_history), time=end_time-start_time))
            print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
                % (np.mean(psnr), np.mean(ssim), 
                    np.mean(mse), np.mean(nmi), np.mean(dist_)))

        if not os.path.exists(f'./save/{model_name}'):
            os.makedirs(f'./save/{model_name}')
        if save and (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    f'./save/{model_name}/{model_name}_checkpoint_{epoch}.pth.tar')

########################################################
### Training IPT                                     ###
########################################################

def train_ipt(model_name, train_data_loader, valid_data_loader, model, optimizer, criterion, 
                        epochs, eval_per_epochs, device, 
                        grad_clip=None, lr_scheduler=None, save=False):

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        loss_history = []
        with tqdm(train_data_loader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                optimizer.zero_grad()

                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs_pred = model(lr_imgs, 0) 

                loss = criterion(sr_imgs_pred, hr_imgs)
                loss.backward()
                loss_history.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        end_time = time.time()
        loss_train = np.mean(loss_history)
        
        print('Epoch: {0}----Loss {loss:.4f}'
            '----Time: {time} secs'.format(epoch, 
            loss=loss_train, time=end_time-start_time))

        # Do evaluation
        if (epoch + 1) % eval_per_epochs == 0 or epoch==0:
            start_time = time.time()
            model.eval()
            loss_history = []
            psnr, ssim, mse, nmi, dist_= [], [], [], [], []
            loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
            loss_fn_alex.to(device)
            with torch.no_grad():
                with tqdm(valid_data_loader, unit='batch') as vepoch:
                    for batch in vepoch:
                        vepoch.set_description(f"Valid at epoch {epoch}")

                        # Prepare Data
                        lr_imgs, hr_imgs = batch
                        lr_imgs = lr_imgs.to(device)
                        hr_imgs = hr_imgs.to(device) 

                        # Forward the model
                        sr_imgs_pred = model(lr_imgs, 0) 

                        loss = criterion(sr_imgs_pred, hr_imgs)
                        loss_history.append(loss.item())
                        vepoch.set_postfix(loss=loss.item())
                        
                        yimgs = quantize(sr_imgs_pred) # Will be [0, 255]
                        yimgs = sr_imgs_pred.cpu().detach().numpy()
                        gtimgs = hr_imgs.cpu().detach().numpy()

                        dist = loss_fn_alex.forward(sr_imgs_pred, hr_imgs)
                        dist_.append(dist.mean().item())

                        for yimg, gtimg in zip(yimgs, gtimgs):
                            psnr.append(skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=2))
                            ssim.append(skimage.metrics.structural_similarity(yimg, gtimg, channel_axis=0))
                            mse.append(skimage.metrics.mean_squared_error(yimg, gtimg))
                            nmi.append(skimage.metrics.normalized_mutual_information(yimg, gtimg))
                        
            end_time = time.time()
            print('Valid at epoch: {0}----Loss {loss:.4f}'
                '----Time: {time} secs'.format(epoch, 
                loss=np.mean(loss_history), time=end_time-start_time))
            print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
                % (np.mean(psnr), np.mean(ssim), 
                    np.mean(mse), np.mean(nmi), np.mean(dist_)))

        if not os.path.exists(f'./save/{model_name}'):
            os.makedirs(f'./save/{model_name}')
        if save and (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    f'./save/{model_name}/{model_name}_checkpoint_{epoch}.pth.tar')


########################################################
### Training SRResNet                                ###
########################################################

def train_single_model(model_name,
                        train_data_loader, valid_data_loader,
                        model, optimizer, criterion, 
                        epochs, eval_per_epochs, device, 
                        grad_clip=None, lr_scheduler=None, save=False):

    
    for epoch in range(epochs):
        model.train()
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
        loss_train = np.mean(loss_history)
        
        print('Epoch: {0}----Loss {loss:.4f}'
            '----Time: {time} secs'.format(epoch, 
            loss=loss_train, time=end_time-start_time))
        
        # Do evaluation
        if (epoch + 1) % eval_per_epochs == 0:
            start_time = time.time()
            model.eval()
            loss_history = []
            psnr, ssim, mse, nmi, dist_= [], [], [], [], []
            loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
            loss_fn_alex.to(device)
            with torch.no_grad():
                with tqdm(valid_data_loader, unit='batch') as vepoch:
                    for batch in vepoch:
                        vepoch.set_description(f"Valid at epoch {epoch}")

                        # Prepare Data
                        lr_imgs, hr_imgs = batch
                        lr_imgs = lr_imgs.to(device)
                        hr_imgs = hr_imgs.to(device) 

                        # Forward the model
                        sr_imgs_pred = model(lr_imgs) # Will be [-1, 1]

                        loss = criterion(sr_imgs_pred, hr_imgs)
                        loss_history.append(loss.item())
                        vepoch.set_postfix(loss=loss.item())
                            
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
                        
            end_time = time.time()
            print('Valid at epoch: {0}----Loss {loss:.4f}'
                '----Time: {time} secs'.format(epoch, 
                loss=np.mean(loss_history), time=end_time-start_time))
            print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
                % (np.mean(psnr), np.mean(ssim), 
                    np.mean(mse), np.mean(nmi), np.mean(dist_)))

        if not os.path.exists(f'./save/{model_name}'):
            os.makedirs(f'./save/{model_name}')
        if save and (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    f'./save/{model_name}/{model_name}_checkpoint_{epoch}.pth.tar')

########################################################
### Training SRGAN                                   ###
########################################################

def train_generative_adversarial_model(model_name,
        train_data_loader, valid_data_loader, 
        generator, discriminator, loss_computer, 
        content_loss_criterion, adversarial_loss_criterion,
        optimizer_generator, optimizer_discriminator, beta,
        epochs, eval_per_epochs, device, 
        grad_clip=None, lr_scheduler=None, save=False):
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start_time = time.time()
        loss_history = []
        with tqdm(train_data_loader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                '''
                Generator Part
                '''
                optimizer_generator.zero_grad()

                # Prepare Data
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device) 

                # Forward the generator
                sr_imgs_pred = generator(lr_imgs)
                sr_imgs_pred = image_converter(sr_imgs_pred, source='[-1, 1]', target='imagenet-norm')
                hr_imgs = image_converter(hr_imgs, source='[-1, 1]', target='imagenet-norm')

                # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
                sr_imgs_in_vgg_space = loss_computer(sr_imgs_pred)
                hr_imgs_in_vgg_space = loss_computer(hr_imgs).detach()

                # Discriminate super-resolved (SR) images
                sr_discriminated = discriminator(sr_imgs_pred)  # (N)

                # Calculate the Perceptual loss
                content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
                adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
                perceptual_loss = content_loss + beta * adversarial_loss                
                
                perceptual_loss.backward()

                loss_history.append(perceptual_loss.item())
                tepoch.set_postfix(perceptual_loss=perceptual_loss.item())

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 
                        max_norm=grad_clip, norm_type=2.0, error_if_nonfinite=False)
                
                if lr_scheduler:
                    lr_scheduler.step()
                
                optimizer_generator.step()

                '''
                Discriminator Part
                '''
                optimizer_discriminator.zero_grad()

                # Discriminate super-resolution (SR) and high-resolution (HR) images
                hr_discriminated = discriminator(hr_imgs)
                sr_discriminated = discriminator(sr_imgs_pred.detach())
                # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
                # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
                # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
                # See FAQ section in the tutorial

                # Binary Cross-Entropy loss
                adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                                adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))


                adversarial_loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 
                        max_norm=grad_clip, norm_type=2.0, error_if_nonfinite=False)

                # Update discriminator
                optimizer_discriminator.step()

        end_time = time.time()
        loss_train = np.mean(loss_history)
        
        print('Epoch: {0}----Loss {loss:.4f}'
            '----Time: {time} secs'.format(epoch, 
            loss=loss_train, time=end_time-start_time))
        
        # Do evaluation
        if (epoch + 1) % eval_per_epochs == 0:
            start_time = time.time()
            generator.eval()
            loss_history = []
            psnr, ssim, mse, nmi, dist_= [], [], [], [], []
            loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
            loss_fn_alex.to(device)
            with torch.no_grad():
                with tqdm(valid_data_loader, unit='batch') as vepoch:
                    for batch in vepoch:
                        vepoch.set_description(f"Valid at epoch {epoch}")

                        # Prepare Data
                        lr_imgs, hr_imgs = batch
                        lr_imgs = lr_imgs.to(device)
                        hr_imgs = hr_imgs.to(device) 

                        # Forward the model
                        sr_imgs_pred = generator(lr_imgs) # out is [-1, 1]

                        loss = content_loss(sr_imgs_pred, hr_imgs)
                        loss_history.append(loss.item())
                        vepoch.set_postfix(loss=loss.item())

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
                        
            end_time = time.time()
            print('Valid at epoch: {0}----Loss {loss:.4f}'
                '----Time: {time} secs'.format(epoch, 
                loss=np.mean(loss_history), time=end_time-start_time))
            print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
                % (np.mean(psnr), np.mean(ssim), 
                    np.mean(mse), np.mean(nmi), np.mean(dist_)))
        
        if not os.path.exists(f'./save/{model_name}'):
            os.makedirs(f'./save/{model_name}')
        if save and (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_generator': optimizer_generator,
                    'optimizer_discriminator': optimizer_discriminator},
                    f'./save/{model_name}_checkpoint_{epoch}.pth.tar')

if __name__ == '__main__':
    # Set proper network setting
    if basic_configs['model_name'] == 'srresnet':
        model_configs = srresnet_configs
    elif basic_configs['model_name'] == 'srgan':
        model_configs = srgan_configs
    elif basic_configs['model_name'] == 'mae':
        model_configs = None
    elif basic_configs['model_name'] == 'ipt':
        model_configs = args_ipt
    else:
        raise NotImplementedError

    run(basic_configs, model_configs)