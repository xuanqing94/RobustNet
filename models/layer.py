import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self, ).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        if self.std > 1.0e-6:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.resize_(x.size()).normal_(0, self.std)
            x.data += self.buffer
        return x

class BReLU(nn.Module):
    def __init__(self, t):
        super(BReLU, self).__init__()
        assert(t > 0)
        self.t = t

    def forward(self, x):
        return x.clamp(0, self.t)
