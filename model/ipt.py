# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.utils.model_zoo
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from einops import rearrange
import copy
import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import datetime
import time
import imageio

import argparse
import lpips
from dataset import SRSamplingDataset
import skimage
from tqdm import tqdm

from utils import image_converter
from visualize import visualize_sampling, batch_visualize


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
    
##################################################################

##################################################################

def make_model(args, parent=False):
    return ipt(args)

class ipt(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ipt, self).__init__()
        
        self.scale_idx = 0
        
        self.args = args
        
        n_feats = args.n_feats
        kernel_size = 3 
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        self.body = VisionTransformer(img_dim=args.patch_size, patch_dim=args.patch_dim, num_channels=n_feats, embedding_dim=n_feats*args.patch_dim*args.patch_dim, num_heads=args.num_heads, num_layers=args.num_layers, hidden_dim=n_feats*args.patch_dim*args.patch_dim*4, num_queries = args.num_queries, dropout_rate=args.dropout_rate, mlp=args.no_mlp ,pos_every=args.pos_every,no_pos=args.no_pos,no_norm=args.no_norm)

        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ) for s in args.scale
        ])
        

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x,self.scale_idx)
        res += x

        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)

        return x 

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos
        
        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
        
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
               
        if self.mlp==False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x+pos)
            x = self.decoder(x, x, query_pos=query_embed)
        
        
        if self.mlp==False:
            x = self.mlp_head(x) + x
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

##################################################################

##################################################################

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

#######################################################

#######################################################
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            forward_function = self.forward_chop

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
    
    def forward_chop(self, x, shave=12):
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size/2)

        scale = self.scale[self.idx_scale]

        h_cut = (h-padsize)%(int(shave/2))
        w_cut = (w-padsize)%(int(shave/2))

        x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave/2)).transpose(0,2).contiguous()

        x_hw_cut = x[...,(h-padsize):,(w-padsize):]
        y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

        x_h_cut = x[...,(h-padsize):,:]
        x_w_cut = x[...,:,(w-padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        
        x_h_top = x[...,:padsize,:]
        x_w_top = x[...,:,:padsize]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize)
        y_unfold = []

        x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_unfold = torch.cat(y_unfold,dim=0)

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
        
        y[...,:padsize*scale,:] = y_h_top
        y[...,:,:padsize*scale] = y_w_top

        y_unfold = y_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        
        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones, padsize*scale-shave*scale, stride=int(shave/2*scale)),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        
        y_inter = y_inter/divisor

        y[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_inter

        y = torch.cat([y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        y_w_cat = torch.cat([y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        y = torch.cat([y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]],dim=3)
        return y.cuda()
    
    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        
        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0),-1,padsize,padsize)
        x_range = x_h_cut_unfold.size(0)//batchsize + (x_h_cut_unfold.size(0)%batchsize !=0)
        y_h_cut_unfold=[]
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_h_cut_unfold = torch.cat(y_h_cut_unfold,dim=0)
        
        y_h_cut = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
        y_h_cut_unfold = y_h_cut_unfold[...,:,int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale))
        
        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale)),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale)) 
        y_h_cut_inter = y_h_cut_inter/divisor
        
        y_h_cut[...,:,int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_h_cut_inter
        return y_h_cut
        
    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        
        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        
        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0),-1,padsize,padsize)
        x_range = x_w_cut_unfold.size(0)//batchsize + (x_w_cut_unfold.size(0)%batchsize !=0)
        y_w_cut_unfold=[]
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold,dim=0)
        
        y_w_cut = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,padsize*scale), padsize*scale, stride=int(shave/2*scale))
        y_w_cut_unfold = y_w_cut_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),:].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
        
        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale)),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
        y_w_cut_inter = y_w_cut_inter/divisor

        y_w_cut[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),:] = y_w_cut_inter
        return y_w_cut

################################################################

################################################################

args_ipt = {
    # All the args for the IPT
    'debug': False,
    'template': '.',

    # Hardware
    'n_threads': 6,
    'cpu': False,
    'n_GPUs': 1,
    'seed': 1,

    # Data
    'dir_data': '/cache/data/',
    'dir_demo': '../test',
    'data_train': 'DIV2K',
    'data_test': 'DIV2K',
    'data_range': '1-800/801-810',
    'ext': 'sep',
    'scale': 4,
    'patch_size': 48,
    'rgb_range': 255,
    'n_colors': 3,
    'no_augment': False,

    # Model
    'model': 'ipt',
    'n_feats': 64,
    'shift_mean': True,
    'precision': 'single', # 'single', 'half': 'FP precision for test (single | half)'

    # Training
    'reset': False, # reset the training
    'test_every': 1000, # do test per every N batches
    'epochs': 300,
    'batch_size': 16,
    'test_batch_size': 1,
    'crop_batch_size': 64,
    'split_batch': 1, # split the batch into smaller chunks
    'self_ensemble': False, # use self-ensemble method for test
    'test_only': False, # set this option to test the model
    'gan_k': 1,# k value for adversarial loss

    # Optimization
    'lr': 1e-4,
    'decay': 200, # learning rate decay type
    'gamma': 0.5,
    'optimizer': 'ADAM', # (SGD | ADAM | RMSprop)
    'momentum': 0.9,
    'betas': (0.9, 0.999),
    'epsilon': 1e-8, # ADAM epsilon for numerical stability
    'weight_decay': 0,
    'gclip': 0, # gradient clipping threshold (0 = no clipping)

    # Loss
    'loss': '1*L1',
    'skip_threshold': 1e8, # skipping batch that has large error

    # Log
    'save': '/cache/results/ipt/',
    'load': 'file name to load',
    'resume': 0, # resume from specific checkpoint
    'save_models': False,
    'print_every': 100,
    'save_results': False,
    'save_gt': False,

    # Cloud
    'moxfile': 1,
    'data_url': None,
    'train_url': None,
    'pretrain': '', # Path to pretrained model
    'load_query': 0,

    # Transformer
    'patch_dim': 3,
    'num_heads': 12,
    'num_layers': 12,
    'dropout_rate': 0,
    'no_norm': False,
    'freeze_norm': False,
    'post_norm': False,
    'no_mlp': False,
    'pos_every': False,
    'no_pos': False,
    'num_queries': 1,
}

################################################################

################################################################

class Checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img):
    return img.mul(1).clamp(0, 255).round().div(1)

################################################################

################################################################

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
            batch_visualize(img_path, 'ipt', model, device, False, False, True)

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
                                            pin_memory=False)   

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

    main_ipt(args, mode = 'visual', img_path="./test/")