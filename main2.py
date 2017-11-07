#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as tfs
from models import *
from torch.utils.data import DataLoader
import time
import sys

# train one epoch
def train(dataloader, net, loss_f, optimizer):
    net.train()
    beg = time.time()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        vx, vy = Variable(x), Variable(y)
        optimizer.zero_grad()
        output = net(vx)
        lossv = loss_f(output, vy)
        lossv.backward()
        optimizer.step()
        correct += y.eq(torch.max(output.data, 1)[1]).sum()
        total += y.numel()
    run_time = time.time() - beg
    return run_time, correct / total

# test and save
def test(dataloader, net, best_acc, model_out):
    net.eval()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        vx = Variable(x, volatile=True)
        output = net(vx)
        correct += y.eq(torch.max(output.data, 1)[1]).sum()
        total += y.numel()
    acc = correct / total
    if acc > best_acc:
        torch.save(net.state_dict(), model_out)
        return acc, acc
    else:
        return acc, best_acc

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--net', type=str, default=None)
    parser.add_argument('--modelIn', type=str, default=None)
    parser.add_argument('--modelOut', type=str, default=None)
    parser.add_argument('--method', type=str, default="momsgd")
    parser.add_argument('--noise', type=float, default=0.0)
    opt = parser.parse_args()
    print(opt)
    epochs = [80, 60, 40, 20]
    if opt.net is None:
        print("opt.net must be specified")
        exit(-1)
    elif opt.net == "vgg16":
        net = VGG("VGG16", opt.noise, opt.noise)
    elif opt.net == "resnetxt":
        net = ResNeXt29_2x64d(opt.noise)
    else:
        print("Invalid opt.net: {}".format(opt.net))
        exit(-1)
    #net = densenet_cifar()
    #net = GoogLeNet()
    #net = MobileNet(num_classes=100)
    #net = stl10(32)
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
    #net = Test()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    loss_f = nn.CrossEntropyLoss()
    net.cuda()
    loss_f.cuda()
    transform_train = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    data = dst.CIFAR10("/home/xqliu/data/cifar10-py", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("/home/xqliu/data/cifar10-py", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    accumulate = 0
    best_acc = 0
    total_time = 0
    for epoch in epochs:
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=.9, weight_decay=5.0e-4)
        for _ in range(epoch):
            accumulate += 1
            run_time, train_acc = train(dataloader, net, loss_f, optimizer)
            test_acc, best_acc = test(dataloader_test, net, best_acc, opt.modelOut)
            total_time += run_time
            print('[Epoch={}] Time:{:.2f}, Train: {:.5f}, Test: {:.5f}, Best: {:.5f}'.format(accumulate, total_time, train_acc, test_acc, best_acc))
            sys.stdout.flush()
        # reload best model
        net.load_state_dict(torch.load(opt.modelOut))
        net.cuda()
        opt.lr /= 10

if __name__ == "__main__":
   main()
