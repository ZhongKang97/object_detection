from __future__ import print_function
import os
import numpy as np
import time
import argparse
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
import utils.util as util
from layers.modules import MultiBoxLoss
from ssd import build_ssd


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        m.bias.data.zero_()


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    decay = args.gamma ** (sum(step >= np.array(args.schedule)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TO LAUNCH VISDOM: python -m visdom.server -port PORT_ID
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--phase', default='train')
parser.add_argument('--save_folder', default='renew_no_pretrain/', help='Location to save checkpoint models')

# training config
parser.add_argument('--iterations', default=130000, type=int, help='Number of training iterations')
parser.add_argument('--no_pretrain', action='store_true', help='default is using pretrain')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='ssd300_0712_iter_30', type=str, help='Resume from checkpoint')

parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--schedule', default=[80000, 100000, 120000], type=float)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

# model params
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--ssd_dim', default=300, type=int)

# runtime config
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# enable gpu_id, launch in terminal: CUDA_VISIBLE_DEVICES=1,2 python train.py
# parser.add_argument('--gpu_id', default='1', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port_id', default=8097, type=int)
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

args = parser.parse_args()
# args.gpu_id = util._process(args.gpu_id)
args.save_folder = 'result/' + args.save_folder + args.phase
if args.resume:
    args.resume = args.save_folder + '/' + args.resume + '.pth'

if args.cuda and torch.cuda.is_available():
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.visdom:
    # TODO
    import visdom
    vis = visdom.Visdom(port=args.port_id)

if not os.path.exists(args.save_folder):
    util.mkdirs(args.save_folder)


train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
ssd_dim = args.ssd_dim
means = (104, 117, 123)  # for voc
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
max_iter = args.iterations
schedule = args.schedule
ssd_net = build_ssd('train', ssd_dim, num_classes)
print(ssd_net)

optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

if args.resume:
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iteration']
        weights = checkpoint['state_dict']
        try:
            ssd_net.load_state_dict(weights)
        except KeyError:
            weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
            ssd_net.load_state_dict(weights_new)

    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
# if args.resume:
#     print('Resuming training, loading {}...'.format(args.resume))
#     ssd_net.load_weights(args.resume)
else:
    args.start_iter = 0
    if args.no_pretrain:
        print('Train from scratch...')
        ssd_net.apply(weights_init)
    else:
        vgg_weights = torch.load('data/pretrain/' + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print('Initializing weights of the newly added layers...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

# write options and loss file
log_file_name = os.path.join(args.save_folder,
                             'opt_loss_{:s}_start_iter_{:d}.txt'.format(args.phase, args.start_iter))
with open(log_file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))

    log_file.write('-------------- End ----------------\n')
    now = time.strftime("%c")
    log_file.write('================ Training Loss (%s) ================\n' % now)

# only shown in console
print('dataset path: %s' % VOCroot)

if args.cuda:
    ssd_net = torch.nn.DataParallel(ssd_net).cuda()
    cudnn.benchmark = True


def train():

    print('Loading Dataset...')
    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())
    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)

    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):

        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        adjust_learning_rate(optimizer, iteration)
        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        t0 = time.time()
        # forward
        out = ssd_net(images)
        # backward
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        # jot down the loss
        if iteration % 5 == 0:

            msg = 'iter %d || Loss: %.4f || time: %.4f sec/iter' % (iteration, loss.data[0], (t1 - t0))
            print(msg)
            with open(log_file_name, "a") as log_file:
                log_file.write('%s\n' % msg)

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                vis.image(images.data[random_batch_index].cpu().numpy())

        # save results
        if iteration % 10 == 0:
            print('Saving state, iter:', iteration)
            torch.save({
                'state_dict': ssd_net.state_dict(),
                'iteration': iteration,
            }, '%s/ssd%d_0712_iter_' % (args.save_folder, args.ssd_dim) + repr(iteration) + '.pth')

    print('Training done.')
    torch.save({
                'state_dict': ssd_net.state_dict(),
                'iteration': iteration,
            }, args.save_folder + '/final_' + args.version + '.pth')


if __name__ == '__main__':
    train()
