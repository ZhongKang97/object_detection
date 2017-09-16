import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data.config import *
import os
import collections
import utils.util as util


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, opts, phase, num_classes, base, extras, head):
        super(SSD, self).__init__()
        self.opts = opts
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox, by the original author
        # init the priors (anchors) based on config
        if opts.prior_config == 'v2':
            self.priorbox = PriorBox(v2, opts.ssd_dim)
        elif opts.prior_config == 'v3':
            self.priorbox = PriorBox(v3, opts.ssd_dim)
        elif opts.prior_config == 'v2_512':
            self.priorbox = PriorBox(v2_512, opts.ssd_dim)

        # for ssd300, priors: [8732 x 4] boxes/anchors
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = Detect(num_classes, 0, opts.top_k, opts.conf_thresh, opts.nms_thresh)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                       # loc preds
                self.softmax(conf.view(-1, self.num_classes)),      # conf preds
                self.priors.type(type(x.data))                      # default boxes
            )
        elif self.phase == "train":
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        # DEPRECATED
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            # self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            weights = torch.load(base_file, map_location=lambda storage, loc: storage)
            weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
            self.load_state_dict(weights_new)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_weight_new(self):
        if self.opts.resume:
            if os.path.isfile(self.opts.resume):
                print(("=> loading checkpoint '{}'".format(self.opts.resume)))
                checkpoint = torch.load(self.opts.resume)
                self.opts.start_iter = checkpoint['iteration']
                weights = checkpoint['state_dict']
                try:
                    self.load_state_dict(weights)
                except KeyError:
                    weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
                    self.load_state_dict(weights_new)
            else:
                print(("=> no checkpoint found at '{}'".format(self.opts.resume)))
        else:
            self.opts.start_iter = 0
            if self.opts.no_pretrain:
                print('Train from scratch...')
                self.apply(util.weights_init)
            else:
                vgg_weights = torch.load('data/pretrain/' + self.opts.basenet)
                print('Loading base network...')
                self.vgg.load_state_dict(vgg_weights)
                print('Initializing weights of the newly added layers...')
                # initialize newly added layers' weights with xavier method
                self.extras.apply(util.weights_init)
                self.loc.apply(util.weights_init)
                self.conf.apply(util.weights_init)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4],
}


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(opts, phase, size, num_classes):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    vgg_layers = vgg(base[str(size)], 3)
    extra_layers = add_extras(extras[str(size)], 1024)
    return SSD(opts, phase, num_classes,
               *multibox(vgg_layers, extra_layers, mbox[str(size)], num_classes))
