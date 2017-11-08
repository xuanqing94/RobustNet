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
