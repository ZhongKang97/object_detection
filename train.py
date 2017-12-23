from __future__ import print_function
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

from data import detection_collate
from data.create_dset import create_dataset
from layers.modules import MultiBoxLoss
from ssd import build_ssd           # this is a function
from option.train_opt import TrainOptions
from utils.util import *
from utils.train import *
from utils.visualizer import Visualizer

# config
option = TrainOptions()
option.setup_config()
args = option.opt

# dataset
dataset = create_dataset(args)
data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                              shuffle=True, collate_fn=detection_collate, pin_memory=True)
# model
ssd_net, (args.start_epoch, args.start_iter) = build_ssd(args, dataset.num_classes)

# init log file
show_jot_opt(args)

# optim, loss
optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(dataset.num_classes, 0.5, True,
                         0, True, 3, 0.5, False, args.use_cuda)
# init visualizer
visual = Visualizer(args)


epoch_size = len(dataset) // args.batch_size
for epoch in range(args.start_epoch, args.max_epoch):

    batch_iterator = iter(data_loader)

    old_lr = optimizer.param_groups[0]['lr']
    adjust_learning_rate(optimizer, epoch, args)
    new_lr = optimizer.param_groups[0]['lr']
    if epoch == args.start_epoch:  # only execute once
        print_log('\ninit learning rate {:f} at epoch {:d}, iter {:d}\n'.format(
            old_lr, epoch, args.start_iter), args.file_name)
        start_iter = args.start_iter
    else:
        start_iter = 0
    if old_lr != new_lr:
        print_log('\nchange learning rate from {:f} to {:f} at epoch {:d}\n'.format(
            old_lr, new_lr, epoch), args.file_name)

    cnt = 0
    for iter_ind in range(start_iter, epoch_size):

        # load train data
        images, targets = next(batch_iterator)
        if args.use_cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]

        # verbose
        cnt += images.size(0)
        if iter_ind == epoch_size - 1:
            print_log('total {:d} images are processed; true total images are {:d}\n'.format(
                cnt, len(dataset)), args.file_name)

        t0 = time.time()
        # forward
        out = ssd_net(images)
        # backward
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets, args.debug_mode)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        losses, progress = (loss, loss_l, loss_c), (epoch, iter_ind, epoch_size)
        # jot down the loss
        if (iter_ind % args.loss_freq) == 0 or iter_ind == (epoch_size-1):

            visual.print_loss(losses, progress, (t0, t1))
            visual.plot_loss(losses, progress)
            visual.print_info(progress, (True, new_lr, t1-t0))

        # save results just in debug mode
        if (iter_ind % args.save_freq == 0) and args.debug_mode:
            save_model(progress, args, (ssd_net, dataset))

    # one epoch ends, save results
    # by default the debug mode won't go here
    if epoch % args.save_freq == 0 or epoch == args.max_epoch:
        save_model(progress, args, (ssd_net, dataset))

print_log('Training done. start_epoch / end_epoch: {:d}/{:d}'.format(
    args.start_epoch, args.max_epoch), args.file_name)
visual.print_info(progress, (False, new_lr, t1-t0))
