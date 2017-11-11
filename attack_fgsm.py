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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def attack_fgsm(input_v, label_v, net, loss, eps, TARGETED = False):
    input_v.requires_grad = True
    adverse = torch.FloatTensor(input_v.size()).zero_().cuda()
    adverse_v = Variable(adverse, requires_grad= True)
    outputs = net(input_v)
    #softmax = nn.Softmax()
    #outputs = softmax(outputs)
    #print(outputs)
    error = loss(outputs,label_v)
    error.backward()
    
    grad = torch.sign(input_v.grad.data)
    #adverse_v.data = torch.clamp(input_v.data + epsilon* grad, 0, 1)
    adverse_v.data = input_v.data + eps * grad
    return adverse_v

def iter_attack_fgsm(input_v, label_v, net, loss, alpha, eps, TARGETED= False):
    #input_v.requires_grad = True
    adverse = input_v.data.clone()
    adverse_v = Variable(adverse, requires_grad= True)
    for _ in range(30):
        outputs = net(adverse_v)
        #softmax = nn.Softmax()
        #outputs = softmax(outputs)
        #print(outputs)
        error = loss(outputs,label_v)
        error.backward()
        normed_grad = alpha * torch.sign(adverse_v.grad.data)
        step_adv = adverse_v.data + normed_grad
        adv = torch.clamp(step_adv - input_v.data, -eps , eps)
        result = torch.clamp(input_v.data + adv, 0, 1)
        adverse_v.data = result
    return adverse_v 

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
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0)
    opt = parser.parse_args()

    net = VGG("VGG16", opt.noise)
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
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    data = dst.CIFAR10("/home/luinx/data/cifar10-py", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("/home/luinx/data/cifar10-py", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=100, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=100, shuffle=True, num_workers=2)
    count, count2 = 0, 0
    for input, output in dataloader_test:
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        adverse_v = attack_fgsm(input_v, label_v, net, loss_f, eps=opt.eps)
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        count += torch.sum(label_v.eq(idx)).data[0]
        count2 += torch.sum(label_v.eq(idx2)).data[0]
        print("Count: {}, Count2: {}".format(count, count2))

        adverse_v.data = adverse_v.data * std_t + mean_t
        input_v.data = input_v.data * std_t + mean_t
        adverse_np = adverse_v.cpu().data.numpy().swapaxes(1, 3)
        input_np = input_v.cpu().data.numpy().swapaxes(1, 3)
        plt.subplot(121)
        plt.imshow(np.abs(input_np[0, :, :, :].squeeze()))
        plt.subplot(122)
        plt.imshow(np.abs(adverse_np[0, :, :, :].squeeze()))
        plt.show()

    print("Accuracy: {}, Attach: {}".format(count / len(data), count2 / len(data)))
