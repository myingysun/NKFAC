'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms


from torch.optim import lr_scheduler
import os
import argparse
from models import *


import sys 
sys.path.append('../')
 
#import optimizers
from NKFAC import NKFAC

torch.distributed.init_process_group(backend="nccl")

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--bs', default=128, type=int, help='batchsize')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--alg', default='nkfac', type=str, help='algorithm')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--path', default='logout/result', type=str, help='path')
parser.add_argument('--model', default='r18', type=str, help='model')
parser.add_argument('--gpug', default=1, type=int, help='gpugroup')
parser.add_argument('--stat_decay', default=0.95, type=float, help='weight decay')
parser.add_argument('--d', default=0.01, type=float, help='weight decay')
parser.add_argument('--tconv', default=100, type=int, help='batchsize')
parser.add_argument('--tinv', default=1000, type=int, help='batchsize')
parser.add_argument('--K', default=1, type=int, help='NEWTONSTEP')
parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for distribute training (default: nccl)')
    # Set automatically by torch distributed launch
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
args = parser.parse_args()

epochs=args.epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
args.local_rank=torch.distributed.get_rank()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
rank = int(os.environ["RANK"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('rank = {}, world_size = {}, device_ids = {}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size(),
            args.local_rank))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
  ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
  ])
  
trainset = torchvision.datasets.CIFAR100(root='/home/sunying/data/cifar-100-python', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True,batch_size=int(args.bs/torch.distributed.get_world_size()), num_workers=4,drop_last=True,sampler=train_sampler)

testset = torchvision.datasets.CIFAR100(root='/home/sunying/data/cifar-100-python', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=4)



# Model
print('==> Building model..')

Num_classes = 100

if args.model=='r18':
    net = ResNet18(Num_classes=Num_classes)
if args.model=='r50':
    net = ResNet50(Num_classes=Num_classes)
if args.model=='v11':
    net = VGG('VGG11',Num_classes=Num_classes)
if args.model=='v19':
    net = VGG('VGG19',Num_classes=Num_classes)
if args.model == 'ggnet':
    net = GoogLeNet()
if args.model=='d121':
    net = DenseNet121(Num_classes=Num_classes)

    
device='cuda'
if device == 'cuda':
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank])
    cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
criterion = nn.CrossEntropyLoss()

#optimizer
WD=args.wd
print('==> choose optimizer..')

if args.alg == 'nkfac':
    optimizer = NKFAC(net,
                      lr=args.lr,
                      momentum=0.9,
                      stat_decay=args.stat_decay,
                      damping1=args.d,
                      damping2=args.d,
                      weight_decay=WD,
                      TCov=args.tconv,
                      TInv=args.tinv,
                      K = args.K)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)   ### milestone lr decay
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.001)    ## cos lr decay

# Training
def train(epoch,net,optimizer):
    if rank == 0: 
      print('\nEpoch: %d' % epoch)
    trainloader.sampler.set_epoch(epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if rank == 0:    
      print('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),correct/total))
    acc=100.*correct/total
    return acc
    
# Testing
def test(epoch,net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if rank == 0: 
      print('Testing:Loss: {:.4f} | Acc: {:.4f}'.format(test_loss/(batch_idx+1),correct/total) )
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and rank == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc

for epoch in range(start_epoch, start_epoch+epochs):
    train_acc=train(epoch,net,optimizer)
    exp_lr_scheduler.step()
    val_acc=test(epoch,net)
print('Best_acc:',best_acc)
