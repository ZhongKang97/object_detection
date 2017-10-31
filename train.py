from __future__ import print_function
import os
import numpy as np
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

from data import detection_collate
from data.create_dset import create_dataset
from layers.modules import MultiBoxLoss
from ssd import build_ssd           # this is a function
from option.train_opt import args


# TODO: LAUNCH VISDOM: python -m visdom.server -port PORT_ID
if args.visdom:
    import visdom
    vis = visdom.Visdom(port=args.port_id)


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    schedule_list = np.array(args.schedule)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


dataset = create_dataset(args)
data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                              shuffle=True, collate_fn=detection_collate, pin_memory=True)

ssd_net = build_ssd(args, args.phase, args.ssd_dim, dataset.num_classes)
optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(dataset.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

ssd_net.load_weight_new()
epoch_size = len(dataset) // args.batch_size
batch_iterator = None
start_iter = ssd_net.opts.start_iter

# write options and loss file
log_file_name = os.path.join(args.save_folder,
                             'opt_loss_{:s}_start_iter_{:d}_end_iter_{:d}.txt'.format(
                                 args.phase, start_iter, args.max_iter))
with open(log_file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))

    log_file.write('-------------- End ----------------\n')
    now = time.strftime("%c")
    log_file.write('================ Training Loss (%s) ================\n' % now)


for iteration in range(start_iter, args.max_iter+1):

    if (not batch_iterator) or (iteration % epoch_size == 0):
        # create batch iterator after one epoch ends
        batch_iterator = iter(data_loader)

    old_lr = optimizer.param_groups[0]['lr']
    adjust_learning_rate(optimizer, iteration)
    new_lr = optimizer.param_groups[0]['lr']
    if iteration == start_iter:
        print('\ninit learning rate {:f} at iter {:d}\n'.format(old_lr, iteration))
    if old_lr != new_lr:
        print('\nchange learning rate from {:f} to {:f} at iter {:d}\n'.format(old_lr, new_lr, iteration))
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
    loss_l, loss_c = criterion(out, targets, args.debug)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()

    # jot down the loss
    if iteration % args.loss_freq == 0:

        msg = '[%s]\titer %d || Loss: %.4f || time: %.4f sec/iter' % \
              (args.experiment_name, iteration, loss.data[0], (t1 - t0))
        print(msg)
        with open(log_file_name, "a") as log_file:
            log_file.write('%s\n' % msg)

        if args.visdom and args.send_images_to_visdom:
            random_batch_index = np.random.randint(images.size(0))
            vis.image(images.data[random_batch_index].cpu().numpy())

    # save results
    if (iteration % args.save_freq == 0) | (iteration == args.max_iter):
        print('Saving state, iter:', iteration)
        torch.save({
            'state_dict': ssd_net.state_dict(),
            'iteration': iteration,
        }, '%s/ssd%d_0712_iter_' % (args.save_folder, args.ssd_dim) + repr(iteration) + '.pth')

print('Training done. start_iter / end_iter: {:d}/{:d}'.format(start_iter, args.max_iter))
torch.save({
            'state_dict': ssd_net.state_dict(),
            'iteration': iteration,
        }, args.save_folder + '/final_' + args.version + '.pth')
