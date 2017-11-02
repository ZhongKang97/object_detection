from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.utils.data as data
import time

from data.create_dset import create_dataset
from option.train_opt import args   # for cifar we also has test here

import torch.backends.cudnn as cudnn
import torch.nn as nn
import layers.from_wyang.models.cifar as models
from utils.from_wyang import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import shutil

use_cuda = torch.cuda.is_available()
show_freq = 100
state = {k: v for k, v in args._get_kwargs()}
# TODO: LAUNCH VISDOM: python -m visdom.server -port PORT_ID
if args.visdom:
    import visdom
    vis = visdom.Visdom(port=args.port_id)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule_cifar:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Progressing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % show_freq == 0 or batch_idx == len(trainloader)-1:
            print('TRAIN\t({:3d}/{:3d})\t\tData: {:.3f}s | Batch: {:.3f}s | '
                  'Loss: {:.4f} | top1: {: .4f} | top5: {:.4f}'.format(
                    batch_idx + 1, len(trainloader), data_time.avg, batch_time.avg,
                    losses.avg, top1.avg, top5.avg))
    #     # plot progress
    #     bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ' \
    #                  'Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | ' \
    #                  'top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                     batch=batch_idx + 1,
    #                     size=len(trainloader),
    #                     data=data_time.avg,
    #                     bt=batch_time.avg,
    #                     # total=bar.elapsed_td,
    #                     # eta=bar.eta_td,
    #                     loss=losses.avg,
    #                     top1=top1.avg,
    #                     top5=top5.avg)
    #     bar.next()
    # bar.finish()
    return losses.avg, top1.avg, top5.avg


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % show_freq == 0 or batch_idx == len(testloader)-1:
            print('TEST\t({:3d}/{:3d})\t\tData: {:.3f}s | Batch: {:.3f}s '
                  ' | Loss: {:.4f} | top1: {: .4f} | top5: {: .4f}'.format(
                    batch_idx + 1, len(testloader),
                    data_time.avg, batch_time.avg,
                    losses.avg, top1.avg, top5.avg))
    #     # plot progress
    #     bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} ' \
    #                  '| ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                     batch=batch_idx + 1,
    #                     size=len(testloader),
    #                     data=data_time.avg,
    #                     bt=batch_time.avg,
    #                     total=bar.elapsed_td,
    #                     eta=bar.eta_td,
    #                     loss=losses.avg,
    #                     top1=top1.avg,
    #                     top5=top5.avg)
    #     bar.next()
    # bar.finish()
    return losses.avg, top1.avg, top5.avg


test_dset = create_dataset(args, 'test')
test_loader = data.DataLoader(test_dset, args.train_batch, num_workers=args.num_workers, shuffle=False)
train_dset = create_dataset(args, 'train')
train_loader = data.DataLoader(train_dset, args.test_batch, num_workers=args.num_workers, shuffle=True)

model = models.__dict__['resnet'](num_classes=train_dset.num_classes, depth=50)
if use_cuda and args.deploy:
    print(args.schedule_cifar)
    model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

title = 'cifar-10-resnet'
logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss',
                  'Train Acc.', 'Test Acc.', 'Train Acc5.', 'Test Acc5.'])

best_acc = 0  # best test accuracy
# train and val/test
for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    train_loss, train_acc, train_acc5 = \
        train(train_loader, model, criterion, optimizer, epoch, True)
    if epoch > 140:
        test_loss, test_acc, test_acc5 = \
            test(test_loader, model, criterion, epoch, True)
    else:
        test_loss, test_acc, test_acc5 = 'n/a', 'n/a', 'n/a'

    # append logger file
    logger.append([str(epoch), state['lr'], train_loss, test_loss,
                   train_acc, test_acc, train_acc5, test_acc5])

    # save model
    test_acc = 0 if test_acc == 'n/a' else test_acc
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch':        epoch + 1,
        'state_dict':   model.state_dict(),
        'acc':          test_acc,
        'best_acc':     best_acc,
        'optimizer':    optimizer.state_dict(),
    }, is_best, checkpoint=args.checkpoint)

logger.close()
logger.plot()
savefig(os.path.join(args.checkpoint, 'log.eps'))

print('Best acc:')
print(best_acc)



