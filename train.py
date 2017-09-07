from __future__ import print_function
import os
import numpy as np
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data

from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
import utils.util as util
from layers.modules import MultiBoxLoss
from ssd import build_ssd


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TO LAUNCH VISDOM: python -m visdom.server -port PORT_ID
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--phase', default='train')
parser.add_argument('--no_pretrain', action='store_false')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--gpu_id', default='0,1', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
# parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')

parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port_id', default=8097, type=int)
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

parser.add_argument('--save_folder', default='renew_no_pretrain/', help='Location to save checkpoint models')     # weights/
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--ssd_dim', default=300, type=int)
args = parser.parse_args()
args.gpu_id = util._process(args.gpu_id)
args.save_folder = 'result/train/' + args.save_folder

if args.cuda and torch.cuda.is_available():
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
# accum_batch_size = 32
# iter_size = accum_batch_size / batch_size
# cfg = (v1, v2)[args.version == 'v2']
max_iter = args.iterations
stepvalues = (80000, 100000, 120000)
ssd_net = build_ssd('train', ssd_dim, num_classes)

log_file_name = os.path.join(args.save_folder, 'opt_loss_{:s}.txt'.format(args.phase))
with open(log_file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))

    log_file.write('-------------- End ----------------\n')
    now = time.strftime("%c")
    log_file.write('================ Training Loss (%s) ================\n' % now)

# only shown in console
print('dataset path: %s' % VOCroot)
print(ssd_net)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    if args.no_pretrain:
        print('Train from scratch...')
        ssd_net.apply(weights_init)
    else:
        vgg_weights = torch.load('weights/' + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print('Initializing weights of the newly added layers...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

if args.cuda:
    # TODO check here
    # torch.cuda.set_device(args.gpu_id[0])
    if len(args.gpu_id) > 1:
        ssd_net = torch.nn.DataParallel(ssd_net, device_ids=args.gpu_id).cuda()
    else:
        ssd_net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    ssd_net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    # if args.visdom:
    #     # initialize visdom loss plot
    #     lot = vis.line(
    #         X=torch.zeros((1,)).cpu(),
    #         Y=torch.zeros((1, 3)).cpu(),
    #         opts=dict(
    #             xlabel='Iteration',
    #             ylabel='Loss',
    #             title='Current SSD Training Loss',
    #             legend=['Loc Loss', 'Conf Loss', 'Loss']
    #         )
    #     )
    #     epoch_lot = vis.line(
    #         X=torch.zeros((1,)).cpu(),
    #         Y=torch.zeros((1, 3)).cpu(),
    #         opts=dict(
    #             xlabel='Epoch',
    #             ylabel='Loss',
    #             title='Epoch SSD Training Loss',
    #             legend=['Loc Loss', 'Conf Loss', 'Loss']
    #         )
    #     )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):

        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                vis.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                                    loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    opts=dict(
                        xlabel='Epoch',
                        ylabel='Loss',
                        title='Epoch SSD Training Loss',
                        legend=['Loc Loss', 'Conf Loss', 'Loss']),
                    win=args.display_id,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        # forward
        t0 = time.time()
        out = ssd_net(images)
        # backward
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        # jot down the loss
        if iteration % 10 == 0:
            msg2 = 'iter %d || Loss: %.4f || time: %.4f sec/iter' % (iteration, loss.data[0], (t1 - t0))
            print(msg2)
            with open(log_file_name, "a") as log_file:
                log_file.write('%s\n' % msg2)

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                vis.image(images.data[random_batch_index].cpu().numpy())

        if args.visdom:
            vis.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                opts=dict(
                    xlabel='Iteration',
                    ylabel='Loss',
                    title='Current SSD Training Loss',
                    legend=['Loc Loss', 'Conf Loss', 'Loss']),
                win=args.display_id+1,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                vis.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    opts=dict(
                        xlabel='Epoch',
                        ylabel='Loss',
                        title='Epoch SSD Training Loss',
                        legend=['Loc Loss', 'Conf Loss', 'Loss']),
                    win=args.display_id,
                    update=True
                )
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), '%s/ssd%d_0712_iter_' % (args.save_folder, args.ssd_dim) +
                       repr(iteration) + '.pth')

    torch.save(ssd_net.state_dict(), args.save_folder + '' + 'final_' + args.version + '.pth')


if __name__ == '__main__':
    train()
