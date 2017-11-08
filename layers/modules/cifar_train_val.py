from utils.from_wyang import AverageMeter, accuracy
from utils.util import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    ExponentialLR, MultiStepLR, StepLR, LambdaLR


def set_lr_schedule(optimizer, plan, others=None):
    if plan == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max')
    elif plan == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=others['milestones'],
                                gamma=others['gamma'])
    return scheduler


def adjust_learning_rate(optimizer, step, args):
    """
    Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Input: step/epoch
    """
    try:
        schedule_list = np.array(args.schedule)
    except AttributeError:
        schedule_list = np.array(args.schedule_cifar)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_weights(model_path, model):
    print('test only mode, loading weights ...')
    checkpoints = torch.load(model_path)
    try:
        print('best test accu is: {:.4f}\n'.format(checkpoints['best_test_acc']))
    except KeyError:
        print('best test accu is: {:.4f}\n'.format(checkpoints['best_acc']))
    weights = checkpoints['state_dict']
    try:
        model.load_state_dict(weights)
    except KeyError:
        weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
        model.load_state_dict(weights_new)
    return model


def remove_batch(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            remove(os.path.join(dir, f))


def save_checkpoint(state, is_best, args, epoch):

    filepath = os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(epoch+1))
    if (epoch+1) % args.save_epoch == 0 or epoch == 0:
        torch.save(state, filepath)
        print_log('model saved at {:s}'.format(filepath), args.file_name)
    if is_best:
        # save the best model
        remove_batch(args.save_folder, 'model_best')
        best_path = os.path.join(args.save_folder, 'model_best_at_epoch_{:d}.pth'.format(epoch+1))
        torch.save(state, best_path)
        print_log('best model saved at {:s}'.format(best_path), args.file_name)


def train(trainloader, model, criterion, optimizer, opt, vis, epoch):

    use_cuda = opt.use_cuda
    structure = opt.model_cifar
    show_freq = opt.show_freq

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
        outputs, _ = model(inputs, targets)  # 128 x 10 x 16
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
        start = time.time()
        optimizer.zero_grad()
        if structure == 'capsule':
            loss.backward(retain_graph=False)
        else:
            loss.backward()
        optimizer.step()
        # print('iter bp time: {:.4f}\n'.format(time.time()-start))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % show_freq == 0 or batch_idx == len(trainloader)-1:
            curr_info = {
                'loss': losses.avg,
                'acc': top1.avg,
                'acc5': top5.avg,
                'data': data_time.avg,
                'batch': batch_time.avg,
            }
            vis.print_loss(curr_info, epoch, batch_idx,
                           len(trainloader), epoch_sum=False, train=True)
            vis.plot_loss(errors=curr_info,
                          epoch=epoch, i=batch_idx, max_i=len(trainloader), train=True)
    return {
        'train_loss': losses.avg,
        'train_acc': top1.avg,
        'train_acc5': top5.avg,
    }


def test(testloader, model, criterion, opt, vis, epoch=0):

    use_cuda = opt.use_cuda
    structure = opt.model_cifar
    show_freq = opt.show_freq

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

        # SHOW histogram here
        # 120.pth, batch_idx = 67
        # TO: only on local MacBook
        which_batch_idx = 67
        # which_batch_idx = 0
        input_vis = vis if opt.draw_hist and (batch_idx == which_batch_idx) else None
        # compute output
        outputs, stats = model(inputs, targets, batch_idx, input_vis)
        if input_vis is not None:
            plot_info = {
                'd2_num': outputs.size(2),
                'curr_iter': batch_idx,
                'model': os.path.basename(opt.cifar_model)
            }
            vis.plot_hist(stats, plot_info)
            a = 1
        if structure == 'capsule':
            outputs = outputs.norm(dim=2)
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
            curr_info = {
                'loss': losses.avg,
                'acc': top1.avg,
                'data': data_time.avg,
                'batch': batch_time.avg,
            }
            vis.print_loss(curr_info, epoch, batch_idx,
                           len(testloader), epoch_sum=False, train=False)
            if opt.test_only is not True:
                vis.plot_loss(errors=curr_info,
                              epoch=epoch, i=batch_idx, max_i=len(testloader), train=False)
    return {
        'test_loss': losses.avg,
        'test_acc': top1.avg,
    }
