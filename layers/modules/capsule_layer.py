# Finger crossed!
import torch.nn as nn


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv