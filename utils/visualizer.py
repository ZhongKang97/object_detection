from utils.util import *
import numpy as np
import ntpath


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        if self.opt.visdom:
            import visdom
            self.vis = visdom.Visdom(port=opt.port, env=opt.experiment_name)
            self.display_win_id = 100

            self.loss_data = {'X': [], 'Y': [], 'legend': ['train_loss']}
            self.acc_data = {'X': [], 'Y': [], 'legend': ['train_acc', 'train_acc5']}
            self.loss_data_test = {'X': [], 'Y': [], 'legend': ['test_loss']}
            self.acc_data_test = {'X': [], 'Y': [], 'legend': ['test_acc']}

    def print_loss(self, errors, epoch, i=0, max_i=0, epoch_sum=False, train=False):

        if self.opt.dataset == 'cifar':
            if epoch_sum:
                print_log('Summary [{:s}]\tepoch [{:d}/{:d}]\t\t'
                          'train_loss: {:.5f}\ttest_loss: {:.5f}\ttrain_acc: {:.5f}\t'
                          'test_acc: {:.5f}\ttrain_acc5: {:.5f}\n'.format(
                            self.opt.experiment_name, epoch, self.opt.max_epoch,
                            errors['train_loss'], errors['test_loss'], errors['train_acc'],
                            errors['test_acc'], errors['train_acc5']), self.opt.file_name)
            else:
                prefix = 'Train' if train else 'Test'
                if train:
                    print_log('{:s} [{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\t'
                              'data: {:.3f}s | batch: {:.3f}s\t'
                              'loss: {:.5f}\tacc: {:.5f}\tacc5: {:.5f}'.format(
                                prefix, self.opt.experiment_name,
                                epoch, self.opt.max_epoch, i, max_i,
                                errors['data'], errors['batch'],
                                errors['loss'], errors['acc'], errors['acc5']), self.opt.file_name)
                else:
                    print_log('{:s} [{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\t'
                              'data: {:.3f}s | batch: {:.3f}s\t'
                              'loss {:.5f}\tacc: {:.5f}'.format(
                                prefix, self.opt.experiment_name,
                                epoch, self.opt.max_epoch, i, max_i,
                                errors['data'], errors['batch'],
                                errors['loss'], errors['acc']), self.opt.file_name)

    def plot_loss(self, errors, epoch, i, max_i, train=False):

        x_progress = epoch + float(i/max_i)

        if train:
            self.loss_data['X'].append(x_progress)
            self.loss_data['Y'].append(errors['loss'])
            self.acc_data['X'].append([x_progress, x_progress])
            self.acc_data['Y'].append([errors['acc'], errors['acc5']])

            self.vis.line(
                X=np.array(self.loss_data['X']),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'title': 'train loss over time',
                    'legend': self.loss_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_win_id
            )
            self.vis.line(
                X=np.array(self.acc_data['X']),
                Y=np.array(self.acc_data['Y']),
                opts={
                    'title': 'train accuracy over time',
                    'legend': self.acc_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'accuracy'},
                win=self.display_win_id + 1
            )
        else:
            self.loss_data_test['X'].append(x_progress)
            self.loss_data_test['Y'].append(errors['loss'])
            self.acc_data_test['X'].append(x_progress)
            self.acc_data_test['Y'].append(errors['acc'])
            self.vis.line(
                X=np.array(self.loss_data_test['X']),
                Y=np.array(self.loss_data_test['Y']),
                opts={
                    'title': 'test loss over time',
                    'legend': self.loss_data_test['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_win_id + 2
            )
            self.vis.line(
                X=np.array(self.acc_data_test['X']),
                Y=np.array(self.acc_data_test['Y']),
                opts={
                    'title': 'test accuracy over time',
                    'legend': self.acc_data_test['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'accuracy'},
                win=self.display_win_id + 3
            )

    def show_image(self, epoch, images):
        # show in the visdom
        idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
        for label, image_numpy in images.items():
            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                           win=self.display_win_id + idx)
            idx += 1

