# Finger crossed!
import math
import torch.nn as nn
from layers.from_wyang.models.cifar.resnet import BasicBlock, Bottleneck
from layers.modules.cap_layer import CapLayer


class CapsNet(nn.Module):
    def __init__(self, depth, route_num, num_classes=100):
        super(CapsNet, self).__init__()

        # ResNet part
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6
        block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        # self.tranfer_conv = nn.ModuleList([
        #         nn.Conv2d(256, 256, kernel_size=3, bias=False),
        #         nn.BatchNorm2d(256), nn.ReLU(True)])
        channel_in = 256 if depth == 50 else 64
        self.tranfer_conv = nn.Conv2d(channel_in, 256, kernel_size=3)  # 256x8x8 -> 256x6x6
        self.cap_layer = CapLayer(num_in_caps=32*6*6, num_out_caps=num_classes,
                                  in_dim=8, out_dim=16,
                                  num_shared=32, route_num=route_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print('linear layer!')
                m.weight.data.normal_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # bs x 256 x 8 x 8
        x = self.tranfer_conv(x)
        x = self.cap_layer(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)