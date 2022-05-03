from model.ipt import Model, Checkpoint, quantize
import argparse
import torch
import lpips
from dataset import SRSamplingDataset
import skimage
import numpy as np
from tqdm import tqdm

from utils import image_converter
from visualize import visualize_sampling

# args = {
#     # All the args for the IPT
#     'debug': False,
#     'template': '.',

#     # Hardware
#     'n_threads': 6,
#     'cpu': False,
#     'n_GPUs': 1,
#     'seed': 1,

#     # Data
#     'dir_data': '/cache/data/',
#     'dir_demo': '../test',
#     'data_train': 'DIV2K',
#     'data_test': 'DIV2K',
#     'data_range': '1-800/801-810',
#     'ext': 'sep',
#     'scale': 4,
#     'patch_size': 48,
#     'rgb_range': 255,
#     'n_colors': 3,
#     'no_augment': False,

#     # Model
#     'model': 'ipt',
#     'n_feats': 64,
#     'shift_mean': True,
#     'precision': 'single', # 'single', 'half': 'FP precision for test (single | half)'

#     # Training
#     'reset': False, # reset the training
#     'test_every': 1000, # do test per every N batches
#     'epochs': 300,
#     'batch_size': 16,
#     'test_batch_size': 1,
#     'crop_batch_size': 64,
#     'split_batch': 1, # split the batch into smaller chunks
#     'self_ensemble': False, # use self-ensemble method for test
#     'test_only': False, # set this option to test the model
#     'gan_k': 1,# k value for adversarial loss

#     # Optimization
#     'lr': 1e-4,
#     'decay': 200, # learning rate decay type
#     'gamma': 0.5,
#     'optimizer': 'ADAM', # (SGD | ADAM | RMSprop)
#     'momentum': 0.9,
#     'betas': (0.9, 0.999),
#     'epsilon': 1e-8, # ADAM epsilon for numerical stability
#     'weight_decay': 0,
#     'gclip': 0, # gradient clipping threshold (0 = no clipping)

#     # Loss
#     'loss': '1*L1',
#     'skip_threshold': 1e8, # skipping batch that has large error

#     # Log
#     'save': '/cache/results/ipt/',
#     'load': 'file name to load',
#     'resume': 0, # resume from specific checkpoint
#     'save_models': False,
#     'print_every': 100,
#     'save_results': False,
#     'save_gt': False,

#     # Cloud
#     'moxfile': 1,
#     'data_url': None,
#     'train_url': None,
#     'pretrain': '', # Path to pretrained model
#     'load_query': 0,

#     # Transformer
#     'patch_dim': 3,
#     'num_heads': 12,
#     'num_layers': 12,
#     'dropout_rate': 0,
#     'no_norm': False,
#     'freeze_norm': False,
#     'post_norm': False,
#     'no_mlp': False,
#     'pos_every': False,
#     'no_pos': False,
#     'num_queries': 1,
# }

# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>



def main_ipt(args, mode, img_path=None):
    checkpoint = Checkpoint(args)
    if checkpoint.ok:
        model = Model(args, checkpoint)
        if args.pretrain == '':
            args.pretrain = "./save/ipt/IPT_sr4.pt"
        state_dict = torch.load(args.pretrain)
        model.model.load_state_dict(state_dict, strict = False)
    
        if mode == 'test':
            test_ipt(args, model)
        elif mode == 'visual':
            model.eval()
            device = torch.device("cpu" if args.cpu else "cuda")
            with torch.no_grad(): # This can significant reduce the size in GPU memory...
                visualize_sampling(img_path, 'ipt', model, device)

def test_ipt(args, model):

    device = torch.device("cuda" if not args.cpu else "cpu")

    dataset = 'set5'
    test_dataset = SRSamplingDataset(dataset,
                        type='test',
                        crop_size=0,
                        scaling_factor=4,
                        lr_img_type='[0, 255]',
                        hr_img_type='[0, 255]')
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                            batch_size=1, 
                            shuffle=True, num_workers=4,
                                            pin_memory=True)   

    model.to(device)
    model.eval()
    psnr, ssim, mse, nmi, dist_= [], [], [], [], []
    loss_fn_alex = lpips.LPIPS(net='alex') # Need [-1, 1]
    loss_fn_alex.to(device)
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as vepoch:
            for batch in vepoch:

                # Prepare Data
                lr_imgs, hr_imgs = batch
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device) 
                # print(torch.cuda.memory_allocated(device=0))

                # Forward the model
                sr_imgs_pred = model(lr_imgs, 0) # This time will be [0, 255] and is float

                yimgs = quantize(sr_imgs_pred) 
                yimgs = yimgs.cpu().detach().numpy() 
                gtimgs = hr_imgs.cpu().detach().numpy()
                    
                dist = loss_fn_alex.forward(sr_imgs_pred, hr_imgs)
                dist_.append(dist.mean().item())
                
                for yimg, gtimg in zip(yimgs, gtimgs):
                    psnr.append(skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=255))
                    ssim.append(skimage.metrics.structural_similarity(yimg, gtimg, channel_axis=0))
                    mse.append(skimage.metrics.mean_squared_error(yimg, gtimg))
                    nmi.append(skimage.metrics.normalized_mutual_information(yimg, gtimg))

    print("Test finished")
    print("PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"
        % (np.mean(psnr), np.mean(ssim), 
            np.mean(mse), np.mean(nmi), np.mean(dist_)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IPT')

    parser.add_argument('--debug', action='store_true',
                        help='Enables debug mode')
    parser.add_argument('--template', default='.',
                        help='You can set various templates in option.py')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=6,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    # Data specifications
    parser.add_argument('--dir_data', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--dir_demo', type=str, default='../test',
                        help='demo image directory')
    parser.add_argument('--data_train', type=str, default='DIV2K',
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='DIV2K',
                        help='test dataset name')
    parser.add_argument('--data_range', type=str, default='1-800/801-810',
                        help='train/test data range')
    parser.add_argument('--ext', type=str, default='sep',
                        help='dataset file extension')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=48,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')

    # Model specifications
    parser.add_argument('--model', default='ipt',
                        help='model name')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # Training specifications
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--crop_batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--split_batch', type=int, default=1,
                        help='split the batch into smaller chunks')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for test')
    parser.add_argument('--test_only', action='store_true',
                        help='set this option to test the model')
    parser.add_argument('--gan_k', type=int, default=1,
                        help='k value for adversarial loss')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=str, default='200',
                        help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--gclip', type=float, default=0,
                        help='gradient clipping threshold (0 = no clipping)')

    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
    parser.add_argument('--skip_threshold', type=float, default='1e8',
                        help='skipping batch that has large error')

    # Log specifications
    parser.add_argument('--save', type=str, default='/cache/results/ipt/',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', action='store_true',
                        help='save output results')
    parser.add_argument('--save_gt', action='store_true',
                        help='save low-resolution and high-resolution images together')

    #cloud
    parser.add_argument('--moxfile', type=int, default=1)
    parser.add_argument('--data_url', type=str,help='path to dataset')
    parser.add_argument('--train_url', type=str, help='train_dir')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--load_query', type=int, default=0)

    #transformer
    parser.add_argument('--patch_dim', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--freeze_norm', action='store_true')
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--pos_every', action='store_true')
    parser.add_argument('--no_pos', action='store_true')
    parser.add_argument('--num_queries', type=int, default=1)

    #denoise
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--sigma', type=float, default=30)

    #derain
    parser.add_argument('--derain', action='store_true')
    parser.add_argument('--derain_test', type=int, default=1)

    #deblur
    parser.add_argument('--deblur', action='store_true')
    parser.add_argument('--deblur_test', type=int, default=1)

    args, unparsed = parser.parse_known_args()

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    args.data_train = args.data_train.split('+')
    args.data_test = args.data_test.split('+')
        
    if args.epochs == 0:
        args.epochs = 1e8

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    main_ipt(args, mode = 'visual', img_path="./test/2.png")