#!/usr/bin/env python3

import sys
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import models

def attack_cw(input_v, label_v, net, c, untarget=True, n_class=10):
    net.eval()
    #net.train()
    index = label_v.data.cpu().view(-1, 1)
    label_onehot = torch.FloatTensor(input_v.size()[0], n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1, index, 1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
    w = 0.5 * torch.log(input_v.data / (1 - input_v.data))
    w_v = Variable(w, requires_grad=True)
    optimizer = optim.Adam([w_v], lr=1.0e-3)
    zero_v = Variable(torch.FloatTensor([0]).cuda(), requires_grad=False)
    for _ in range(300):
        net.zero_grad()
        optimizer.zero_grad()
        adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = (torch.max(torch.mul(output, label_onehot_v), 1)[0])
        other = (torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        error = torch.sum(diff * diff)
        if untarget:
            error += c * torch.sum(torch.max(real - other, zero_v))
        else:
            error += c * torch.sum(torch.max(other - real, zero_v))
        error.backward()
        optimizer.step()
    return adverse_v, diff

def attack_fgsm(input_v, label_v, net, epsilon):
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.data.clone()
    adverse_v = Variable(adverse)
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad.data)
    adverse_v.data += epsilon * grad
    return adverse_v

def attack_rand_fgsm(input_v, label_v, net, epsilon):
    alpha = epsilon / 2
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.data.clone() + alpha * torch.sign(torch.FloatTensor(input_v.data.size()).normal_(0, 1).cuda())
    adverse_v = Variable(adverse)
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad.data)
    adverse_v.data += (epsilon - alpha) * grad
    return adverse_v

#Ensemble by sum of probability
def ensemble_infer(input_v, net, n=50, nclass=10):
    net.eval()
    batch = input_v.size()[0]
    softmax = nn.Softmax()
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    for i in range(n):
        prob += softmax(net(input_v)).data
    _, pred = torch.max(prob, 1)
    return Variable(pred)

def acc_under_attack(dataloader, net, c, attack_f, ensemble=1):
    correct = 0
    tot = 0
    distort = 0.0
    for k, (input, output) in enumerate(dataloader):
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        # attack
        adverse_v, diff = attack_f(input_v, label_v, net, c)
        # defense
        net.eval()
        adverse_v = Variable(adverse_v.data, volatile=True)
        if ensemble == 1:
            _, idx = torch.max(net(adverse_v), 1)
        else:
            idx = ensemble_infer(adverse_v, net, n=ensemble)
        correct += torch.sum(label_v.eq(idx)).data[0]
        tot += output.numel()
        distort += torch.sum(diff.data * diff.data)
        if k >= 15:
            break
    return correct / tot, np.sqrt(distort / tot)

def peek(dataloader, net, src_net, c, attack_f, denormalize_layer):
    count, count2, count3 = 0, 0, 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        input_v, label_v = Variable(x.cuda()), Variable(y.cuda())
        adverse_v = attack_f(input_v, label_v, src_net, c)
        net.eval()
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        idx3 = ensemble_infer2(adverse_v, net)
        count += torch.sum(label_v.eq(idx)).data[0]
        count2 += torch.sum(label_v.eq(idx2)).data[0]
        count3 += torch.sum(label_v.eq(idx3)).data[0]
        less, more = check_in_bound(adverse_v, denormalize_layer)
        print("<0: {}, >1: {}".format(less, more))
        print("Count: {}, Count2: {}, Count3: {}".format(count, count2, count3))
        ok = input("Continue next batch? y/n: ")
        if ok == 'n':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--defense', type=str, required=True)
    parser.add_argument('--modelIn', type=str, required=True)
    parser.add_argument('--c', type=str, default='1.0')
    parser.add_argument('--noiseInit', type=float, default=0)
    parser.add_argument('--noiseInner', type=float, default=0)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--mode', type=str, default='peek')
    parser.add_argument('--ensemble', type=int, default=1)
    opt = parser.parse_args()
    # parse c
    opt.c = [float(c) for c in opt.c.split(',')]
    if opt.mode == 'peek' and len(opt.c) != 1:
        print("When opt.mode == 'peek', then only one 'c' is allowed")
        exit(-1)
    if opt.net == "vgg16" or opt.net == "vgg16-robust":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.vgg.VGG("VGG16")
        elif opt.defense == "brelu":
            net = models.vgg_brelu.VGG("VGG16", 0.0)
        elif opt.defense == "rse":
            net = models.vgg_rse.VGG("VGG16", opt.noiseInit, opt.noiseInner)
    elif opt.net == "resnext":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.resnext.ResNeXt29_2x64d()
        elif opt.defense == "brelu":
            net = models.resnext_brelu.ResNeXt29_2x64d(0)
        elif opt.defense == "rse":
            net = models.resnext_rse.ResNeXt29_2x64d(opt.noiseInit, opt.noiseInner)
    elif opt.net == "stl10_model":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.stl10_model.stl10(32)
        elif opt.defense == "brelu":
            # no noise at testing time
            net = models.stl10_model_brelu.stl10(32, 0.0)
        elif opt.defense == "rse":
            net = models.stl10_model_rse.stl10(32, opt.noiseInit, opt.noiseInner)
    net = nn.DataParallel(net, device_ids=range(1))
    net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()
    loss_f = nn.CrossEntropyLoss()

    if opt.dataset == 'cifar10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])

        transform_test = tfs.Compose([
            tfs.ToTensor(),
            ])
        data_test = dst.CIFAR10(opt.root, download=False, train=False, transform=transform_test)
    elif opt.dataset == 'stl10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(96, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])
        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.STL10(opt.root, split='test', download=False, transform=transform_test)
    else:
        print("Invalid dataset")
        exit(-1)
    assert data_test
    dataloader_test = DataLoader(data_test, batch_size=100, shuffle=False)
    if opt.mode == 'peek':
        peek(dataloader_test, net, src_net, opt.c[0], attack_f, denormalize_layer)
    elif opt.mode == 'test':
        print("#c, test accuracy")
        for c in opt.c:
            acc, avg_distort = acc_under_attack(dataloader_test, net, c, attack_cw, opt.ensemble)
            print("{}, {}, {}".format(c, acc, avg_distort))
            sys.stdout.flush()
