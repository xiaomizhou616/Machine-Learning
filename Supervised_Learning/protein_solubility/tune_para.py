from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from bio_dataset import BioDataset
from network import classifier


parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default='./cleandata_1.csv', help='path to dataset file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./models', help='folder to save model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--InputDim', type=int, default=20, help='Dimension of the inputs')
parser.add_argument('--nLayers', type=int, default=0, help='Number of hidden layers')
parser.add_argument('--nNeurons', type=int, default=20, help='Number of neurons')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_entire_dataset = BioDataset(opt.csv_file, split='train')
assert train_entire_dataset
train_entire_dataloader = torch.utils.data.DataLoader(train_entire_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
#cross validation
val_acc = 0.

for fold in range(10):
    print("fold: {}".format(str(fold)))
    train_dataset = BioDataset(opt.csv_file, split='train', is_entire=False, fold=fold)
    val_dataset = BioDataset(opt.csv_file, split='train', is_entire=False, fold=fold, val=True)
    assert train_dataset
    assert val_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
                                         
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)

    net = classifier(InputDim=opt.InputDim, nLayers=opt.nLayers, nNeurons=opt.nNeurons).to(device)
    print(net)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    
    for epoch in range(opt.niter):
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            net.zero_grad()
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            err = criterion(output, y)
            err.backward()
            optimizer.step()
            
            print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, opt.niter, i, len(train_dataloader),
                 err.item()))
                 
    torch.save(net.state_dict(), '%s/net_fold_%d.pth' % (opt.outf, fold))
    net.eval()
    num_correct = 0
    for i, data in enumerate(val_dataloader, 0):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        output = net(x)
        if opt.cuda:
            num_correct += torch.sum(torch.eq(torch.ge(output, 0.5), y.type(torch.cuda.ByteTensor))).item()
        else:
            num_correct += torch.sum(torch.eq(torch.ge(output, 0.5), y.type(torch.ByteTensor))).item()
    
    val_acc += float(num_correct)/len(val_dataset)

val_acc /= 10
#train on the entire training set
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
net = classifier(InputDim=opt.InputDim, nLayers=opt.nLayers, nNeurons=opt.nNeurons).to(device)
print(net)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
for epoch in range(opt.niter):
    net.train()
    for i, data in enumerate(train_entire_dataloader, 0):
        net.zero_grad()
        x, y = data
        x = x.to(device)
        y = y.to(device)
        output = net(x)
        err = criterion(output, y)
        err.backward()
        optimizer.step()
            
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, opt.niter, i, len(train_entire_dataloader),
                 err.item()))

torch.save(net.state_dict(), '%s/net.pth' % (opt.outf))
net.eval()
num_correct = 0
for i, data in enumerate(train_entire_dataloader, 0):
    x, y = data
    x = x.to(device)
    y = y.to(device)
    output = net(x)
    if opt.cuda:
        num_correct += torch.sum(torch.eq(torch.ge(output, 0.5), y.type(torch.cuda.ByteTensor))).item()
    else:
        num_correct += torch.sum(torch.eq(torch.ge(output, 0.5), y.type(torch.ByteTensor))).item()

train_acc = float(num_correct)/len(train_entire_dataset)
print("cross val acc: {}".format(str(val_acc)))
print("train acc: {}".format(str(train_acc)))