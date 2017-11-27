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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import models

def attack_cw(input_v, label_v, net, c, untarget=True, n_class=10, margin=0):
    index = label_v.data.cpu().view(-1, 1)
    label_onehot = torch.FloatTensor(input_v.size()[0], n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1, index, 1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
    #w = 0.5 * torch.log(input_v.data / (1 - input_v.data)) #+ torch.FloatTensor(input_v.size()).normal_(0, 1).cuda()
    #w_v = Variable(w, requires_grad=True)
    adverse = input_v.data.clone()
    adverse_v = Variable(adverse, requires_grad=True)
    optimizer = optim.Adam([adverse_v], lr=1.0e-2)
    #optimizer = optim.SGD([adverse_v], lr=1.0e-2, momentum=0.9, weight_decay=5e-4)
    margin_v = Variable(torch.FloatTensor([margin]).cuda(), requires_grad=False)
    for i in range(1000):
        net.zero_grad()
        optimizer.zero_grad()
        #adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = (torch.max(output * label_onehot_v, 1)[0])
        other = (torch.max(output * (1-label_onehot_v) - label_onehot_v * 10000, 1)[0])
        error = torch.sum(diff * diff)
        if untarget:
            error += c * torch.sum(torch.max(real - other, margin_v))
            #if real.data[0] < other.data[0]:
            #    break
        else:
            error += c * torch.sum(torch.max(other - real, margin_v))
            #if other.data[0] < real.data[0]:
            #    break
        error.backward()
        optimizer.step()
    return adverse_v, adverse_v - input_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--defense', type=str, required=True)
    parser.add_argument('--modelIn', type=str, required=True)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--noiseInit', type=float, default=0)
    parser.add_argument('--noiseInner', type=float, default=0)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--imgf', type=str, required=True)
    opt = parser.parse_args()
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
        #data = dst.CIFAR10(opt.root, download=False, train=True, transform=transform_train)
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
        #data = dst.STL10(opt.root, split='train', download=False, transform=transform_train)
        data_test = dst.STL10(opt.root, split='test', download=False, transform=transform_test)
    else:
        print("Invalid dataset")
        exit(-1)
    assert data_test
    img, label = data_test[1]
    img = img.cuda()
    img.resize_(1, 3, 32, 32)
    img_v = Variable(img)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()
    output = net(img_v)
    print("Logits: {}".format(output.data))
    output = torch.max(output, dim=1)[1].data[0]
    print("True label: {}, predicted as: {}".format(classes[label], classes[output]))
    for i in range(10):
        if i == label:
            continue
        # target attack as label i
        label_v = Variable(torch.LongTensor([[i]]).cuda())
        attack_img, diff = attack_cw(img_v, label_v, net, opt.c, untarget=False, n_class=10)
        net.eval()
        output = torch.max(net(attack_img), dim=1)[1].data[0]
        distortion = torch.sum(diff.data * diff.data)
        save_image(1.0-(attack_img - img_v).data.cpu(), './{}/{}_{}_{}.png'.format(opt.imgf, opt.defense, classes[label], classes[i]))
        print("Target label: {}, predicted as: {}, distortion: {:.12f} ..... {}".format(classes[i], classes[output], distortion, "\033[32mok\033[0m" if i == output else "\033[31mfail\033[0m"))
