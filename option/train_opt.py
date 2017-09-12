import argparse
from data import VOCroot

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--phase', default='train')
parser.add_argument('--save_folder', default='renew_ssd512_default/', help='Location to save checkpoint models')

# training config
parser.add_argument('--iterations', default=130000, type=int, help='Number of training iterations')
parser.add_argument('--no_pretrain', action='store_true', help='default is using pretrain')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='ssd300_0712_iter_30', type=str, help='Resume from checkpoint')

parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--schedule', default=[80000, 100000, 120000], type=float)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

# model params
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--ssd_dim', default=512, type=int)

# runtime config
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# enable gpu_id, launch in terminal: CUDA_VISIBLE_DEVICES=1,2 python train.py
# parser.add_argument('--gpu_id', default='1', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port_id', default=8097, type=int)
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

args = parser.parse_args()
# args.gpu_id = util._process(args.gpu_id)

args.save_folder = 'result/' + args.save_folder + args.phase
if args.resume:
    args.resume = args.save_folder + '/' + args.resume + '.pth'