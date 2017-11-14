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

def check_in_bound(adverse_v, denormalize_layer):
    denormed = denormalize_layer(adverse_v)
    less = torch.sum(torch.lt(denormed, 0))
    more = torch.sum(torch.ge(denormed, 1))
    return less, more

def attack_cw(input_v, label_v, net, c):
    net.train()
    n_class = len(classes)
    index = label_v.data.cpu().view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
    adverse = input_v.data.clone()
    adverse_v = Variable(adverse, requires_grad=True)
    optimizer = optim.Adam([adverse_v], lr=1.0e-3)
    zero_v = Variable(torch.FloatTensor([0]).cuda(), requires_grad=False)
    for _ in range(300):
        optimizer.zero_grad()
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = (torch.max(torch.mul(output, label_onehot_v), 1)[0])
        other = (torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        error = torch.sum(diff * diff)
        error += c * torch.sum(torch.max(real - other, zero_v))
        #print(error.data[0])
        error.backward()
        optimizer.step()
        #TODO: replace with tanh change of variable
        #adverse_v.data.clamp_(0, 1)
    return adverse_v

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

#Ensemble by voting
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

#Ensemble by sum of probability
def ensemble_infer2(input_v, net, n=50):
    net.eval()
    batch = input_v.size()[0]
    softmax = nn.Softmax()
    nclass = len(classes)
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    for i in range(n):
        prob += softmax(net(input_v)).data
    _, pred = torch.max(prob, 1)
    return Variable(pred)

def acc_under_attack(dataloader, net, src_net, c, attack_f, ensemble=1):
    correct = 0
    tot = 0
    for k, (input, output) in enumerate(dataloader):
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        # attack
        adverse_v = attack_f(input_v, label_v, src_net, c)
        # defense
        net.eval()
        if ensemble == 1:
            _, idx = torch.max(net(adverse_v), 1)
        else:
            idx = ensemble_infer2(adverse_v, net, ensemble)
        correct += torch.sum(label_v.eq(idx)).data[0]
        tot += output.numel()
    return correct / tot

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
    parser.add_argument('--srcModel', required=True)
    parser.add_argument('--dstModel', required=True)
    parser.add_argument('--c', type=str, default='1.0')
    parser.add_argument('--noiseInit', type=float, default=0)
    parser.add_argument('--noiseInner', type=float, default=0)
    parser.add_argument('--root', required=True)
    parser.add_argument('--attack', required=True)
    parser.add_argument('--mode', type=str, default='peek')
    parser.add_argument('--ensemble', type=int, default=1)
    opt = parser.parse_args()
    # parse c
    opt.c = [float(c) for c in opt.c.split(',')]
    if opt.mode == 'peek' and len(opt.c) != 1:
        print("When opt.mode == 'peek', then only one 'c' is allowed")
        exit(-1)
    if opt.attack == 'CW':
        attack_f = attack_cw
    elif opt.attack == 'FGSM':
        attack_f = attack_fgsm
    elif opt.attack == 'RAND_FGSM':
        attack_f = attack_rand_fgsm
    else:
        print('Invalid attacker name')
        exit(-1)
    net = VGG("VGG16", opt.noiseInit, opt.noiseInner)
    src_net = VGG("VGG16", 0, 0)
    net = nn.DataParallel(net, device_ids=range(1))
    src_net = nn.DataParallel(src_net, device_ids=range(1))
    loss_f = nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(opt.dstModel))
    src_net.load_state_dict(torch.load(opt.srcModel))
    net.cuda()
    src_net.cuda()
    loss_f.cuda()
    mean = (0.4914, 0.4822, 0.4465)
    mean_t = torch.FloatTensor(mean).resize_(1, 3, 1, 1).cuda()
    std = (0.2023, 0.1994, 0.2010)
    std_t = torch.FloatTensor(std).resize_(1, 3, 1, 1).cuda()
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
    dataloader_test = DataLoader(data_test, batch_size=100, shuffle=True, num_workers=2)
    if opt.mode == 'peek':
        peek(dataloader_test, net, src_net, opt.c[0], attack_f, denormalize_layer)
    elif opt.mode == 'test':
        print("#c, test accuracy")
        for c in opt.c:
            acc = acc_under_attack(dataloader_test, net, src_net, c, attack_f, opt.ensemble)
            print("{}, {}".format(c, acc))
            sys.stdout.flush()
