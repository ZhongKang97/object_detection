# Finger crossed!
import os
import time
import shutil
import torch
from utils.from_wyang import AverageMeter, accuracy


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def train(trainloader, model,
          criterion, optimizer, use_cuda,
          structure, show_freq):

    FIX_INPUT = False
    has_data = False
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

        if FIX_INPUT:
            if has_data:
                inputs, targets = fix_inputs, fix_targets
            else:
                fix_inputs, fix_targets = inputs, targets
                has_data = True
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)  # 128 x 10 x 16
        if structure == 'capsule':
            outputs = outputs.norm(dim=2)

        # _, ind = outputs[4, :].max(0)
        # print('predict index: {:d}'.format(ind.data[0]))
        # one_sample = outputs[4, :]
        # check = torch.eq(one_sample, one_sample[0])
        # if check.sum().data[0] == len(one_sample):
        #     print('output is the same across all classes: {:.4f}\n'.format(one_sample[0].data[0]))

        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if structure == 'capsule':
            loss.backward(retain_graph=False)
        else:
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
    return losses.avg, top1.avg, top5.avg


def test(testloader, model, criterion, use_cuda,
         show_freq, structure):
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
        if structure == 'capsule':
            outputs = outputs.norm(dim=2)
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
                    data_time.avg, batch_time.avg, losses.avg, top1.avg, top5.avg))
    return losses.avg, top1.avg, top5.avg
