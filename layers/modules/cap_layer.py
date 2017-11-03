# Finger crossed!
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def softmax_dim(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


def squash(vec):
    # vec: 128 x 10 x 16
    norm = vec.norm(dim=2)
    norm_squared = norm ** 2
    coeff = norm_squared / (1 + norm_squared)
    coeff2 = torch.unsqueeze((coeff/norm), dim=2)
    # coeff2: 128 x 10 x 1
    return torch.mul(vec, coeff2)


class CapLayer(nn.Module):
    def __init__(self, num_in_caps, num_out_caps,
                 in_dim, out_dim, num_shared, route_num):
        super(CapLayer, self).__init__()
        self.in_dim = in_dim
        self.num_shared = num_shared
        self.W = [nn.Linear(in_dim, out_dim, bias=True) for _ in range(num_shared)]
        self.b = Variable(torch.zeros(num_out_caps, num_in_caps))  # initial value
        self.route_num = route_num

    def forward(self, input):
        bs, in_channels, h, w = input.size()
        assert in_channels == self.num_shared * self.in_dim

        b = self.b.expand(bs, self.b.size(0), self.b.size(1))  # expand along batch dim

        input = input.view(bs, self.num_shared, -1, self.in_dim)
        groups = input.chunk(self.num_shared, dim=1)
        u = [group.chunk(h * w, dim=2) for group in groups]
        # self.W[1](u[1][0]), u[i][j], i = 1...32, j = 1...36
        pred = [self.W[i](in_vec) for i, group in enumerate(u) for in_vec in group]
        # pred, list[1152], each entry, Variable, size [128x1x1x16]
        pred = torch.stack(pred).permute(1, 0, 2, 3, 4).squeeze()  # \hat(s)_j -> 128, 1152, 16

        for i in range(self.route_num):
            c = softmax_dim(b, axis=1)              # 128 x 10 x 1152, c_nji, \sum_j = 1
            s = torch.matmul(c, pred)               # 128 x 10 x 16
            v = squash(s)
            delta_b = torch.matmul(pred, v.permute(0, 2, 1)).permute(0, 2, 1)
            b = torch.add(b, delta_b)
        return v


class MarginLoss(nn.Module):

    def __init__(self, num_classes, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
        # norm = output.norm(dim=2)        # 128 x 10
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        zero = Variable(torch.zeros(1))
        pos_part = torch.max(zero, self.pos - output).pow(2)
        neg_part = torch.max(zero, output - self.neg).pow(2)
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return loss.sum() / output.size(0)
