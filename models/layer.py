import torch
from torch.autograd import Variable
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        if self.std > 1.0e-6:
            if self.buffer is None:
                self.buffer = Variable(torch.Tensor(x.size()).normal_(0, self.std).cuda(), requires_grad=False)
            else:
                self.buffer.data.resize_(x.size()).normal_(0, self.std)
            return x + self.buffer
        return x

class BReLU(nn.Module):
    def __init__(self, t):
        super(BReLU, self).__init__()
        assert(t > 0)
        self.t = t

    def forward(self, x):
        return x.clamp(0, self.t)

class Normalize(nn.Module):
    def __init__(self, mean_vec, std_vec):
        super(Normalize, self).__init__()
        self.mean = Variable(mean_vec.view(1, 3, 1, 1), requires_grad=False)
        self.std = Variable(std_vec.view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        # x: (batch, 3, H, W)
        # mean, std: (1, 3, 1, 1)
        return (x - self.mean) / self.std
        #return x

class DeNormalize(nn.Module):
    def __init__(self, mean_vec, std_vec):
        super(DeNormalize, self).__init__()
        self.mean = Variable(mean_vec.view(1, 3, 1, 1), requires_grad=False)
        self.std = Variable(std_vec.view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        return x * self.std + self.mean
        #return x
