import os
import time
from tqdm import tqdm
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
data_dir = './data'

save = True
batch_size = 16
epochs = 100
num_workers = 4
eval_per_epochs = 5
learning_rate = 1e-4
learning_schedule = True
grad_clip = False

def run():
    pass

def train(train_data_loader, model, optimizer):

    model.train()

    start_time = time.time()
    
    for i, batch in tqdm(train_data_loader):
        pass