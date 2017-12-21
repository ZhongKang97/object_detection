from utils.util import *
import torch


def adjust_learning_rate(optimizer, step, args):
    """
        Sets the learning rate to the initial LR decayed by gamma
        at every specified step/epoch

        Adapted from PyTorch Imagenet example:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py

        step could also be epoch
    """
    schedule_list = np.array(args.schedule)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_model(progress, args, others):

    epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
    model, dataset = others[0], others[1]
    prefix = 'debug_' if args.debug_mode else ''
    print_log('Saving state at epoch/iter [{:d}/{:d}][{:d}/{:d}] ...'.format(
        epoch, args.max_epoch, iter_ind, epoch_size), args.file_name)
    torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'iter': iter_ind,
    }, '{:s}/{:s}ssd{:d}_{:s}_epoch_{:d}_iter_{:d}.pth'.format(
        args.save_folder, prefix, args.ssd_dim, dataset.name, epoch, iter_ind))
