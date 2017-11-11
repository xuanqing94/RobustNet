#!/usr/bin/env python3

from numba import jit

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

#@jit(nopython=True)
#batch_size: number of coordinate used in optimization
def coordinate_ADAM(losses, indice, grad,  batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2] - losses[i*2+1]) / 0.0002 
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def attack(input_v, label_v, net, c, batch_size= 256, TARGETED=False):
    n_class = len(classes)
    index = label_v.data.cpu().view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
	#print(label_onehot.scatter)
    var_size = input_v.view(-1).size()[0]
    #print(var_size)
    real_modifier = torch.FloatTensor(input_v.size()).zero_().cuda()
    for _ in range(1000): 
        random_set = np.random.permutation(var_size)
        losses = np.zeros(2*batch_size, dtype=np.float32)
        #print(torch.sum(real_modifier))
        for i in range(2*batch_size):
            modifier = real_modifier.clone().view(-1)
            if i%2==0:
                modifier[random_set[i//2]] += 0.0001 
            else:
                modifier[random_set[i//2]] -= 0.0001
            modifier = modifier.view(input_v.size())
            modifier_v = Variable(modifier, requires_grad=True).cuda()
            output = net(input_v + modifier_v)
            real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
            other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
            loss1 = torch.sum(modifier_v*modifier_v)
            if TARGETED:
                loss2 = c* torch.sum(torch.clamp(other - real, min=0))
            else:
                loss2 = c* torch.sum(torch.clamp(real - other, min=0))
            error = loss1 + loss2
            losses[i] = error.data[0]
        print(np.sum(losses))
        if loss2.data[0]==0:
            break
        grad = np.zeros(batch_size, dtype=np.float32)
        mt = np.zeros(var_size, dtype=np.float32)
        vt = np.zeros(var_size, dtype=np.float32)
        adam_epoch = np.ones(var_size, dtype = np.int32)
        np_modifier = real_modifier.cpu().numpy()
        lr = 0.01
        beta1, beta2 = 0.9, 0.999
        #for i in range(batch_size):
        coordinate_ADAM(losses, random_set[:batch_size], grad, batch_size, mt, vt, np_modifier, lr, adam_epoch, beta1, beta2)
        real_modifier = torch.from_numpy(np_modifier)
    real_modifier_v = Variable(real_modifier, requires_grad=True).cuda()
    
    return input_v + real_modifier_v

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
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=2)
    count, count2 = 0, 0
    net.eval()
    for input, output in dataloader_test:
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        adverse_v = attack(input_v, label_v, net, opt.c)
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        count = torch.sum(label_v.eq(idx)).data[0]
        count2 = torch.sum(label_v.eq(idx2)).data[0]
        print("Count: {} Count2: {}".format(count, count2))

        adverse_v.data = adverse_v.data * std_t + mean_t
        input_v.data = input_v.data * std_t + mean_t
        adverse_np = adverse_v.cpu().data.numpy().swapaxes(1, 3)
        input_np = input_v.cpu().data.numpy().swapaxes(1, 3)
        plt.subplot(121)
        plt.imshow(np.abs(input_np[0, :, :, :].squeeze()))
        plt.subplot(122)
        plt.imshow(np.abs(adverse_np[0, :, :, :].squeeze()))
        plt.show()

    #print("Accuracy: {}, Attach: {}".format(count / len(data), count2 / len(data)))
