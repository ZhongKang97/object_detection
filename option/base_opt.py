import argparse
from utils.util import *
import random


class BaseOptions(object):

    def initialize(self):
        self.parser = argparse.ArgumentParser(description='Object Detection')
        self.parser.add_argument('--experiment_name', default='ssd_rerun')
        self.parser.add_argument('--dataset', default='coco', help='[ voc|coco ]')
        self.parser.add_argument('--debug_mode', default=True, type=str2bool)
        self.parser.add_argument('--base_save_folder', default='result')

        self.parser.add_argument('--manual_seed', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--no_visdom', action='store_true')
        self.parser.add_argument('--port_id', default=8090, type=int)

        # model params
        self.parser.add_argument('--ssd_dim', default=512, type=int)
        self.parser.add_argument('--prior_config', default='v2_512', type=str)

    def setup_config(self):

        self.opt.save_folder = os.path.join(self.opt.base_save_folder,
                                            self.opt.experiment_name, self.opt.phase)
        if not os.path.exists(self.opt.save_folder):
            mkdirs(self.opt.save_folder)

        seed = random.randint(1, 10000) if self.opt.manual_seed == -1 else self.opt.manual_seed
        self.opt.random_seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.opt.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.opt.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.opt.phase == 'train':
            if self.opt.debug_mode:
                self.opt.loss_freq = 10
                self.opt.save_freq = self.opt.loss_freq
            else:
                self.opt.loss_freq = 200    # in iter unit
                self.opt.save_freq = 5      # in epoch unit
        else:
            # TODO: test stage
            pass


