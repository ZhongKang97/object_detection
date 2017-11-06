from utils.util import *
import numpy as np
import ntpath


class Visualizer(object):
    def __init__(self, opt):

        self.opt = opt
        if self.opt.visdom:
            import visdom
            self.vis = visdom.Visdom(port=opt.port, env=opt.experiment_name)
            self.display_win_id = 1

    def print_loss(self, epoch, i, max_i, errors):

        if self.opt.model == 'default':
            if self.opt.recurrent_loss:
                print_log('[{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\tloss_D (real/fake) {:.5f} ({:.5f}/{:.5f})\t'
                          'loss_SEG (ske/G || ske_rec/enc_rec) {:.5f} ({:.5f}/{:.5f} || {:.5f}/{:.5f})'.format(
                            self.opt.experiment_name, epoch, self.opt.max_epoch, i, max_i,
                            errors['loss_D'], errors['loss_D_real'], errors['loss_D_fake'],
                            errors['loss_S_E_G'], errors['loss_skeleton'], errors['loss_G'],
                            errors['loss_ske_rec'], errors['loss_enc_rec']), self.opt.file_name)
            else:
                print_log('[{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\tloss_D (real/fake) {:.5f} ({:.5f}/{:.5f})\t'
                          'loss_SEG (ske/G) {:.5f} ({:.5f}/{:.5f})'.format(self.opt.experiment_name,
                                                                           epoch, self.opt.max_epoch, i, max_i,
                                                                           errors['loss_D'], errors['loss_D_real'],
                                                                           errors['loss_D_fake'], errors['loss_S_E_G'],
                                                                           errors['loss_skeleton'], errors['loss_G']),
                          self.opt.file_name)

    def plot_loss(self, epoch, i, max_i, errors):

        if (not hasattr(self, 'plot_data_1')) and (not hasattr(self, 'plot_data_2')):
            temp_ = list(errors.keys())
            if self.opt.model == 'default' or self.opt.add_gan_loss:
                # with gan loss, we need to split the losses
                loss_D_list = temp_[0:3] if self.opt.model == 'default' else temp_[-4:]
                loss_rest = temp_[3:] if self.opt.model == 'default' else temp_[0:5]
                self.plot_data_1 = {'X': [], 'Y': [], 'legend': loss_D_list}
                self.plot_data_2 = {'X': [], 'Y': [], 'legend': loss_rest}

            # elif (self.opt.model == 'encoder' or self.opt.model == 'vae' or self.opt.model == 'ae') \
            #         and (not self.opt.add_gan_loss):
            else:
                self.plot_data_1 = {'X': [], 'Y': [], 'legend': temp_}

        x_progress = (epoch-1) + float(i/max_i)
        self.plot_data_1['X'].append(x_progress)
        self.plot_data_1['Y'].append([errors[k] for k in self.plot_data_1['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data_1['X'])]*len(self.plot_data_1['legend']), 1),
            Y=np.array(self.plot_data_1['Y']),
            opts={
                'title': self.opt.experiment_name + ' loss over time',
                'legend': self.plot_data_1['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_win_id
        )
        if self.opt.model == 'default' or self.opt.add_gan_loss:
            self.plot_data_2['X'].append(x_progress)
            self.plot_data_2['Y'].append([errors[k] for k in self.plot_data_2['legend']])
            self.vis.line(
                X=np.stack([np.array(self.plot_data_2['X'])]*len(self.plot_data_2['legend']), 1),
                Y=np.array(self.plot_data_2['Y']),
                opts={
                    'title': self.opt.experiment_name + ' loss over time',
                    'legend': self.plot_data_2['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_win_id_more
            )

    def show_image(self, epoch, images):
        # show in the visdom
        idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
        for label, image_numpy in images.items():
            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                           win=self.display_win_id + idx)
            idx += 1

