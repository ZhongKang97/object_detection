# Finger crossed!
import math
import torch.nn as nn
from layers.from_wyang.models.cifar.resnet import BasicBlock, Bottleneck
from layers.modules.cap_layer import CapLayer, CapLayer2, squash
import time


class CapsNet(nn.Module):
    def __init__(self, depth, opts,
                 num_classes=100, structure='capsule'):
        super(CapsNet, self).__init__()

        # ResNet part
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6
        block = Bottleneck if depth >= 44 else BasicBlock
        if hasattr(opts, 'cap_model'):
            self.cap_model = opts.cap_model
        else:
            self.cap_model = 'v0'
        self.cap_N = opts.cap_N
        self.structure = structure
        self.inplanes = 16
        self.skip_pre_squash = opts.skip_pre_squash
        self.skip_pre_transfer = opts.skip_pre_transfer

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        channel_in = 256 if depth == 50 else 64
        self.tranfer_conv = nn.Conv2d(channel_in, 256, kernel_size=3)  # 256x8x8 -> 256x6x6
        self.tranfer_bn = nn.BatchNorm2d(256)
        self.tranfer_relu = nn.ReLU(True)
        # ablation study here
        self.avgpool = nn.AvgPool2d(6)
        self.fc = nn.Linear(256, num_classes)
        # capsule module
        self.cap_layer = CapLayer(opts, num_in_caps=32*6*6, num_out_caps=num_classes,
                                  in_dim=8, out_dim=16,
                                  num_shared=32)
        if self.cap_model == 'v1' or self.cap_model == 'v2' or self.cap_model == 'v5':
            self.cap_dim_1 = 16
            # transfer convolution to capsule
            self.transfer1_1 = nn.Sequential(*[
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                # nn.BatchNorm2d(int(16*self.cap_dim)),
                # nn.ReLU(True)
            ])
            self.cap1 = CapLayer2(16, self.cap_dim_1, 32, 16,
                                  route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
            # transfer capsule to convolution
            self.transfer1_2 = nn.Sequential(*[
                nn.ConvTranspose2d(self.cap_dim_1, 16, kernel_size=3,
                                   padding=1, stride=2, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            ])

            self.cap_dim_2 = 32
            self.transfer2_1 = nn.Sequential(*[
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
            ])
            self.cap2 = CapLayer2(32, self.cap_dim_2, 16, 8,
                                  route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
            self.transfer2_2 = nn.Sequential(*[
                nn.ConvTranspose2d(self.cap_dim_2, 32, kernel_size=3,
                                   padding=1, stride=2, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            ])

            self.cap_dim_3 = 64
            self.transfer3_1 = nn.Sequential(*[
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ])
            self.cap3 = CapLayer2(64, self.cap_dim_3, 8, 4,
                                  route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
            self.transfer3_2 = nn.Sequential(*[
                nn.ConvTranspose2d(self.cap_dim_3, 64, kernel_size=3,
                                   padding=1, stride=2, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ])
        if self.cap_model == 'v2' or self.cap_model == 'v5':
            self.cap_dim_4 = 64
            self.cap4 = CapLayer2(64, self.cap_dim_4, 4, 10, as_final_output=True,
                                  route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
        # THERE IS NO VERSION 3
        if self.cap_model == 'v4':
            self.cap_v4_dim = 64
            self.transfer_v4 = nn.Sequential(*[
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ])
            self.cap_v4_1 = CapLayer2(64, self.cap_v4_dim, 8, 8,
                                      route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
            self.cap_v4_2 = CapLayer2(self.cap_v4_dim, self.cap_v4_dim, 8, 10, as_final_output=True,
                                      route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
        if self.cap_model == 'v5':
            self.cap_v5_dim = 64
            self.long_cap = CapLayer2(self.cap_v5_dim, self.cap_v5_dim, 4, 4,
                                      route_num=opts.route_num, b_init=opts.b_init, w_version=opts.w_version)
            self.bucket = nn.Sequential(*[
                nn.Conv2d(self.cap_v5_dim, self.cap_v5_dim, kernel_size=1),
                nn.BatchNorm2d(self.cap_v5_dim),
                nn.ReLU(True)
            ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # print('linear layer!')
                m.weight.data.normal_()
                m.bias.data.zero_()
        # print('passed init')

    def forward(self, x, target=None, curr_iter=0, vis=None):
        # start = time.time()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)                  # 16 x 32 x 32
        # if self.cap_model == 'v1':
        #     x = self.transfer1_1(x)
        #     # do squash first
        #     x = self._do_squash(x)
        #     x = self.cap1(x)
        #     x = self.transfer1_2(x)
        x = self.layer2(x)                  # 32 x 16 x 16
        # if self.cap_model == 'v1':
        #     x = self.transfer2_1(x)
        #     x = self._do_squash(x)
        #     x = self.cap2(x)
        #     x = self.transfer2_2(x)
        x = self.layer3(x)                  # bs x 64(for depth=20) x 8 x 8
        if self.cap_model == 'v1' or \
                self.cap_model == 'v2' or \
                self.cap_model == 'v5':

            if not self.skip_pre_transfer:
                x = self.transfer3_1(x)
            if not self.skip_pre_squash:
                x = self._do_squash(x)
            x = self.cap3(x)                # bs, 64, 4, 4
            if self.cap_model == 'v1':
                x = self.transfer3_2(x)     # bs, 64, 8, 8

        if self.cap_model == 'v2':
            x = self.cap4(x)
        elif self.cap_model == 'v4':
            if not self.skip_pre_transfer:
                x = self.transfer_v4(x)
            if not self.skip_pre_squash:
                x = self._do_squash(x)
            x = self.cap_v4_1(x)
            x = self.cap_v4_2(x)
        elif self.cap_model == 'v5':
            for i in range(self.cap_N):
                x = self.long_cap(x)
                x = self.bucket(x)
            x = self.cap4(x)   # use the head of v2
        else:
            # v1, v0, capsule_original, baseline, etc.

            x = self.tranfer_conv(x)
            x = self.tranfer_bn(x)
            x = self.tranfer_relu(x)        # bs x 64 x 6 x 6
            if self.structure == 'capsule':
                # print('conv time: {:.4f}'.format(time.time() - start))
                start = time.time()
                x, stats = self.cap_layer(x, target, curr_iter, vis)
                # print('last cap total time: {:.4f}'.format(time.time() - start))
            elif self.structure == 'resnet':
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
        return x, stats

    def _do_squash(self, x):
        spatial_size = x.size(2)
        input_channel = x.size(1)
        x = x.resize(x.size(0), x.size(1), int(spatial_size**2)).permute(0, 2, 1)
        x = squash(x)
        x = x.permute(0, 2, 1).resize(x.size(0), input_channel, spatial_size, spatial_size)
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