from utils.util import *
import numpy as np
import ntpath


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        if self.opt.no_visdom is False:
            import visdom
            name = opt.experiment_name
            self.vis = visdom.Visdom(port=opt.port_id, env=name)
            # loss/line 100, text 200, images/hist/etc 300
            self.dis_win_id_line = 100
            self.dis_win_id_txt = 200
            self.dis_win_id_im = 300
            self.loss_data = {'X': [], 'Y': [], 'legend': ['total_loss', 'loss_c', 'loss_l']}

    def plot_loss(self, errors, progress, others=None):

        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        x_progress = epoch + float(iter_ind/epoch_size)

        self.loss_data['X'].append([x_progress, x_progress, x_progress])
        self.loss_data['Y'].append([loss, loss_l, loss_c])

        self.vis.line(
            X=np.array(self.loss_data['X']),
            Y=np.array(self.loss_data['Y']),
            opts={
                'title': 'Train loss over time',
                'legend': self.loss_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.dis_win_id_line
        )

    def print_loss(self, errors, progress, others=None):

        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        t0, t1 = others[0], others[1]
        msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
              'Loss: {:.4f}, loc: {:.4f}, cls: {:.4f} ||\t' \
              'Time: {:.4f} sec/image'.format(
                self.opt.experiment_name, epoch, self.opt.max_epoch, iter_ind, epoch_size,
                loss, loss_l, loss_c, (t1 - t0)/self.opt.batch_size)
        print_log(msg, self.opt.file_name)

    def show_image(self, epoch, images):
        # TODO: visualize image here
        # show in the visdom
        idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
        for label, image_numpy in images.items():
            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                           win=self.display_win_id + idx)
            idx += 1
        # if args.use_visdom:
        #     random_batch_index = np.random.randint(images.size(0))
        #     args.vis.image(images.data[random_batch_index].cpu().numpy())

    def print_info(self, progress, others):

        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        still_run, lr = others[0], others[1]

        status = 'RUNNING' if still_run else 'DONE'
        dynamic = 'curr lr {:f}<br/>' \
                  'epoch/iter [{:d}/{:d}][{:d}/{:d}]<br/>' \
                  'est. left time: {:f}<br/>'.format(
                    lr,
                    epoch, self.opt.max_epoch, iter_ind, epoch_size,
                    0)
        common_suffix = '<br/><br/>-----------<br/>batch_size: {:d}'.format(self.opt.batch_size)

        msg = 'status: <b>{:s}</b><br/>'.format(status) + dynamic + common_suffix
        self.vis.text(msg, win=self.dis_win_id_txt)
