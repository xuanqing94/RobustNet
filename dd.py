#!/usr/bin/env python3
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--t', default=1, type = float, help='temperature')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/luinx/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/luinx/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = VGG('VGG16', 0)
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    net.load_state_dict(torch.load('./vgg16/noise0_IV.pth'))
else:
    print('==> Building model..')
    net = VGG('VGG16', 0)
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

dd_net = VGG('VGG16', 0)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(1))
    cudnn.benchmark = True
    dd_net.cuda()
    dd_net = torch.nn.DataParallel(dd_net, device_ids=range(1))
   

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dd_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



def cross_entropy( input_v, target):
    log_input = torch.log(input_v)
    product = torch.mul(log_input, target)
    loss = torch.sum(product)
    loss *= -1/input_v.size()[0]
    return loss


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
# dd-Training
def dd_train(epoch):
    t = args.t
    print('\nEpoch: %d' % epoch)
    net.eval()
    for i in net.parameters():
        i.requires_grad = False
    dd_net.train()
    for j in dd_net.parameters():
        j.requires_grad = True
    train_loss = 0
    correct = 0
    total = 0
    m = nn.Softmax()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs,  targets = inputs.cuda(),  targets.cuda()
        optimizer.zero_grad()
        inputs, targets= Variable(inputs), Variable(targets)
        dd_outputs = dd_net(inputs)
        dd_outputs_t = m(dd_outputs/t)
        outputs = net(inputs)
        outputs_t = m(outputs/t)
        #print(dd_outputs.data[0, :])
        #print(outputs.data[0, :])
        loss = cross_entropy(dd_outputs_t, outputs_t)
       # print(loss.data[0],loss.requires_grad)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(dd_outputs.data, 1)
#        _, predicted2 = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
#        correct2 = predicted2.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' 
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

lr = args.lr
mt = 0.9
for epoch in range(start_epoch, start_epoch+50):
    dd_train(epoch)
    if epoch//10==0:
        lr *= 0.95
        mt *= 0.5
        optimizer = optim.SGD(dd_net.parameters(), lr= lr, momentum=0.5, weight_decay=5e-4)
    #test(epoch)
torch.save(dd_net.state.dict(), "./dd_net.pth")
