#!/usr/bin/env python3

import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import *
import numpy as np
from scipy.stats import mode

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def attack(input_v, label_v, net, c, normalize_layer, denormalize_layer, targeted):
    net.train()
    n_class = len(classes)
    index = label_v.data.cpu().view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
    adverse = input_v.data.clone()
    adverse_v = Variable(adverse, requires_grad=True)
    optimizer = optim.Adam([adverse_v], lr=0.001)
    zero_v = Variable(torch.FloatTensor([0]).cuda(), requires_grad=False)
    for _ in range(300):
        optimizer.zero_grad()
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = (torch.max(torch.mul(output, label_onehot_v), 1)[0])
        other = (torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        error = c * torch.sum(diff * diff)
        if targeted:
            error += torch.sum(torch.max(other - real, zero_v))
        else:
            error += torch.sum(torch.max(real - other, zero_v))
        print(error.data[0])
        error.backward()
        optimizer.step()
        #TODO: replace with tanh change of variable
        #adverse_v.data.clamp_(0, 1)
    return adverse_v

def ensemble_infer(input_v, net, n=20):
    net.eval()
    batch = input_v.size()[0]
    buff = np.zeros((batch, n))
    for i in range(n):
        _, idx = torch.max(net(input_v), 1)
        buff[:, i] = idx.data.cpu().numpy()
    voting, _ = mode(buff, axis=1)
    voting = voting.squeeze()
    return Variable(torch.from_numpy(voting).long().cuda())

def acc_under_attack(dataloader, net, c, targeted=False):
    correct = 0
    tot = 0
    for input, output in dataloader:
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        adverse_v = attack(input_v, label_v, net, opt.c)
        net.eval()
        _, idx = torch.max(net(adverse_v), 1)
        correct += torch.sum(label_v.eq(idx)).data[0]
        tot += output.numel()
    return correct / tot

def peek(dataloader, net, c, normalize_layer, denormalize_layer, targeted=False):
    count, count2, count3 = 0, 0, 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        input_v, label_v = Variable(x.cuda()), Variable(y.cuda())
        adverse_v = attack(input_v, label_v, net, opt.c, normalize_layer, denormalize_layer, targeted)
        net.eval()
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        idx3 = ensemble_infer(adverse_v, net)
        count += torch.sum(label_v.eq(idx)).data[0]
        count2 += torch.sum(label_v.eq(idx2)).data[0]
        count3 += torch.sum(label_v.eq(idx3)).data[0]
        print("Count: {}, Count2: {}, Count3: {}".format(count, count2, count3))
        ok = input("Continue next batch? y/n: ")
        if ok == 'n':
            break

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelIn', type=str, default=None)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--noiseInit', type=float, default=0)
    parser.add_argument('--noiseInner', type=float, default=0)
    parser.add_argument('--root', type=str, default=None)
    opt = parser.parse_args()

    if opt.root is None:
        print("opt.root must be specified")
        exit(-1)
    net = VGG("VGG16", opt.noiseInit, opt.noiseInner)
    net = nn.DataParallel(net, device_ids=range(1))
    loss_f = nn.CrossEntropyLoss()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()
    loss_f.cuda()
    mean = (0.4914, 0.4822, 0.4465)
    mean_t = torch.FloatTensor(mean).resize_(1, 3, 1, 1).cuda()
    std = (0.2023, 0.1994, 0.2010)
    std_t = torch.FloatTensor(std).resize_(1, 3, 1, 1).cuda()
    normalize_layer = Normalize(mean_t, std_t)
    denormalize_layer = DeNormalize(mean_t, std_t)
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean, std),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean, std),
        ])
    data = dst.CIFAR10(opt.root, download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10(opt.root, download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=100, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=100, num_workers=2)
    peek(dataloader_test, net, opt.c, normalize_layer, denormalize_layer, False)
