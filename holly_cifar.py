from __future__ import print_function
import torch.optim as optim
import torch.utils.data as data
from data.create_dset import create_dataset
from option.train_opt import args   # for cifar we also has test here
import torch.backends.cudnn as cudnn
import torch.nn as nn
import layers.from_wyang.models.cifar as models
from utils.from_wyang import Logger, savefig
from layers.modules.capsule import CapsNet
from layers.modules.cap_layer import MarginLoss
from layers.modules.cifar_train_val import *
from utils.visualizer import Visualizer

use_cuda = torch.cuda.is_available()
show_freq = 10
state = {k: v for k, v in args._get_kwargs()}
vis = Visualizer(args)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule_cifar:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


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

# model = models.__dict__['resnet'](num_classes=train_dset.num_classes, depth=50)
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
if args.use_CE_loss:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = MarginLoss(num_classes=10) \
        if args.model_cifar == 'capsule' else nn.CrossEntropyLoss()

if use_cuda:
    criterion = criterion.cuda()
    if args.deploy:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
cudnn.benchmark = True

title = 'cifar-10-resnet'
logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss',
                  'Train Acc.', 'Test Acc.', 'Train Acc5.', 'Test Acc5.'])

best_acc = 0  # best test accuracy
# train and test
if args.test_only:
    test_loss, test_acc, test_acc5 = \
            test(test_loader, model, criterion, use_cuda,
                 structure=args.model_cifar, show_freq=show_freq)
    print('test acc is {:.4f}'.format(test_acc))
else:
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc, train_acc5 = \
            train(train_loader, model, criterion,
                  optimizer, use_cuda,
                  structure=args.model_cifar, show_freq=show_freq)

        if epoch > 139:
            test_loss, test_acc, test_acc5 = \
                test(test_loader, model, criterion, use_cuda,
                     structure=args.model_cifar, show_freq=show_freq)
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
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('Best acc:')
    print(best_acc)






