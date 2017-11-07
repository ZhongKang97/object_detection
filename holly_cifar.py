from __future__ import print_function
import torch.optim as optim
import torch.utils.data as data
from data.create_dset import create_dataset
import torch.backends.cudnn as cudnn
import torch.nn as nn
from layers.modules.capsule import CapsNet
from layers.modules.cap_layer import MarginLoss
from layers.modules.cifar_train_val import *
from utils.visualizer import Visualizer
from utils.util import *
from option.train_opt import args   # for cifar we also has test here

args.show_freq = 20
args.show_test_after_epoch = 139  # -1
args = show_jot_opt(args)
vis = Visualizer(args)


def adjust_learning_rate(optimizer, step):
    """
    Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Input: step or epoch
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

test_dset = create_dataset(args, 'test')
test_loader = data.DataLoader(test_dset, args.test_batch,
                              num_workers=args.num_workers, shuffle=False)
train_dset = create_dataset(args, 'train')
train_loader = data.DataLoader(train_dset, args.train_batch,
                               num_workers=args.num_workers, shuffle=True)

model = CapsNet(depth=20, num_classes=10,
                opts=args, structure=args.model_cifar)
if args.test_only:
    model = load_weights(args, model)

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
if args.use_CE_loss:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = MarginLoss(num_classes=10) \
        if args.model_cifar == 'capsule' else nn.CrossEntropyLoss()

if args.use_cuda:
    criterion = criterion.cuda()
    if args.deploy:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
cudnn.benchmark = True

best_acc = 0  # best test accuracy
# train and test
if args.test_only:
    info = test(test_loader, model, criterion, args, vis)
    print('test acc is {:.4f}'.format(info['test_acc']))
else:
    for epoch in range(args.epochs):

        old_lr = optimizer.param_groups[0]['lr']
        adjust_learning_rate(optimizer, epoch)
        new_lr = optimizer.param_groups[0]['lr']
        if epoch == args.start_epoch - 1:
            print_log('\ninit learning rate {:f} at iter {:d}\n'.format(
                old_lr, epoch), args.file_name)
        if old_lr != new_lr:
            print_log('\nchange learning rate from {:f} to '
                      '{:f} at iter {:d}\n'.format(old_lr, new_lr, epoch), args.file_name)
        info = train(train_loader, model, criterion, optimizer, args, vis, epoch)

        if epoch > args.show_test_after_epoch:
            extra_info = test(test_loader, model, criterion, args, vis, epoch)
        else:
            extra_info = dict()
            extra_info['test_loss'], extra_info['test_acc'] = 0, 0

        # show loss in console and log into file
        info.update(extra_info)
        vis.print_loss(info, epoch, epoch_sum=True)

        # save model
        test_acc = 0 if extra_info['test_acc'] == 'n/a' else extra_info['test_acc']
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch':        epoch+1,
            'state_dict':   model.state_dict(),
            'test_acc':          test_acc,
            'best_test_acc':     best_acc,
            'optimizer':    optimizer.state_dict(),
        }, is_best, args, epoch)
        msg = 'status: <b>RUNNING</b><br/>curr best test acc {:.4f}'.format(best_acc)
        vis.vis.text(msg, win=200)

    print_log('Best acc: {:.f}. Training done.'.format(best_acc), args.file_name)
    msg = 'status: DONE\nbest test acc {:.4f}'.format(best_acc)
    vis.vis.text(msg, win=200)







