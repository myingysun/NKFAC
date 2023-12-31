import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torchvision.models as models
import math
import numpy as np
from torch.optim import lr_scheduler


import sys 
sys.path.append('../')

from NKFAC import NKFAC

torch.distributed.init_process_group(backend="nccl")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1*32/32, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--model', default='r18', type=str, help='model')
parser.add_argument('--path', default='test', type=str, help='model')
parser.add_argument('--alg', default='nkfac', type=str, help='model')
parser.add_argument('--gpug', default=1, type=int, help='gpugroup')
parser.add_argument('--stat_decay', default=0.95, type=float, help='weight decay')
parser.add_argument('--d', default=0.01, type=float, help='weight decay')
parser.add_argument('--tconv', default=20, type=int, help='batchsize')
parser.add_argument('--tinv', default=200, type=int, help='batchsize')

#parser.add_argument('--backend', type=str, default='nccl',
#                        help='backend for distribute training (default: nccl)')
#    # Set automatically by torch distributed launch
parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

rank = int(os.environ["RANK"])  

def main():
    args = parser.parse_args()
    args.local_rank=torch.distributed.get_rank()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    args.world_size = int(os.environ["WORLD_SIZE"])
  
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu


    # create model
    num_classes=1000
    if args.model=='r50':
        model = models.resnet50()
    if args.model=='r18':
        model = models.resnet18()


    #torch.distributed.get_world_size()
    model.cuda()
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int(args.workers / args.world_size)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # choose optimizer
    if args.alg=='sgd': 
       optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay = args.weight_decay)
    if args.alg=='adamw':   
       optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    if args.alg == 'nkfac':
        optimizer = NKFAC(model,
                          lr=args.lr,
                          momentum=0.9,
                          stat_decay=args.stat_decay,
                          damping1=args.d,
                          damping2=args.d,
                          weight_decay=args.weight_decay,
                          TCov=args.tconv,
                          TInv=args.tinv,
                          single_gpu=False)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
         ]))



    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    best_acc1=0
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        exp_lr_scheduler.step()
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1
        best_acc1 = max(acc1, best_acc1)
    end.record()

    torch.cuda.synchronize()
    print('time:',start.elapsed_time(end))

#train
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    total = 0
    train_loss = 0
    correct = 0
    # switch to train mode
    model.train()
    if rank == 0:
        print('\nEpoch: %d' % epoch)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if args.gpu is not None:
         #input = input.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)
        input, target = input.to('cuda'), target.to('cuda')
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        train_loss += loss.item()
        #correct +=acc1[0]
        total += target.size(0)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        #preoptimizer.step()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    if rank == 0:
       #print('Training: Top1: {top1.avg:.4f}|loss:{losses.avg:.4f}'.format(top1=top1, losses=losses))
       print('Training Top1: {top1.avg:.4f}|Top5: {top5.avg:.4f}'.format(top1=top1, top5=top5))
    #print('Training: top1: {:.4f} '.format(correct/total))

# test
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    val_loss = 0
    total = 0
    correct = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            val_loss +=loss.item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            val_loss += loss.item()
        if rank == 0:    
           print('Testing: Top1: {top1.avg:.4f}|Top5: {top5.avg:.4f}'.format(top1=top1, top5=top5))
        #print('Testing: top1: {:.4f} '.format(correct/total))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
