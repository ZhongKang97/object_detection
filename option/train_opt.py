import argparse
from utils.util import *

parser = argparse.ArgumentParser(description='Capsule Object Detection')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--dataset', default='cifar', help='[ voc | coco | cifar ]')
parser.add_argument('--experiment_name', default='cifar_base_104_no_relu')
parser.add_argument('--deploy', action='store_true')
# args_temp = parser.parse_args()

parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

# if args_temp.dataset == 'voc' or args_temp.dataset == 'coco':
# training config
parser.add_argument('--max_iter', default=130000, type=int, help='Number of training iterations')
parser.add_argument('--no_pretrain', action='store_true', help='default is using pretrain')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='ssd300_0712_iter_30', type=str, help='Resume from checkpoint')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--schedule', default=[80000, 100000, 120000], nargs='+')
# model params
parser.add_argument('--ssd_dim', default=512, type=int)
parser.add_argument('--prior_config', default='v2_512', type=str)

# elif args_temp.dataset == 'cifar':
# for cifar only
parser.add_argument('--draw_hist', action='store_true')
parser.add_argument('--test_only', action='store_false')
parser.add_argument('--non_target_j', action='store_true')
# v1 is the newly added capsule network
# parser.add_argument('--cap_model', default='v5', type=str, help='only valid when model_cifar is [capsule]')
parser.add_argument('--model_cifar', default='capsule', type=str, help='resnet | capsule')
parser.add_argument('--cap_N', default=3, type=int, help='for v5 only, parallel N CapLayers')
parser.add_argument('--skip_pre_transfer', action='store_true')
parser.add_argument('--skip_pre_squash', action='store_true')
parser.add_argument('--use_CE_loss', action='store_true')
parser.add_argument('--route_num', default=3, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--schedule_cifar', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--train_batch', default=128, type=int, metavar='N')
parser.add_argument('--test_batch', default=128, type=int, metavar='N')
# see 'cap_layer.py' about the explanations of the following arguments
parser.add_argument('--w_version', default='v2', type=str, help='[v0 | v1, ...]')
parser.add_argument('--look_into_details', action='store_true')
parser.add_argument('--has_relu_in_W', action='store_true')
# squash is much better
parser.add_argument('--do_squash', action='store_true', help='for w_v3 alone')
parser.add_argument('--b_init', default='zero', type=str, help='[zero | rand]')
parser.add_argument('--save_epoch', default=20, type=int)

# runtime and display
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--port', default=4000, type=int)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()

# GENERAL SETTING
# TODO: change ssd iter to epoch training
args.start_epoch = 1
args.max_epoch = args.epochs if hasattr(args, 'epochs') else args.max_iter

args.debug = not args.deploy
args.phase = 'train'
# args.gpu_id = util._process(args.gpu_id)
args.save_folder = os.path.join('result', args.experiment_name, args.phase)

if args.dataset == 'voc' or args.dataset == 'coco':
    if args.resume:
        args.resume = os.path.join(args.save_folder, (args.resume + '.pth'))

    if type(args.schedule[0]) == str:
        temp_ = args.schedule[0].split(',')
        schedule = list()
        for i in range(len(temp_)):
            schedule.append(int(temp_[i]))
        args.schedule = schedule

    if args.debug:
        args.loss_freq, args.save_freq = 1, 5
    else:
        args.loss_freq, args.save_freq = 50, 5000

if args.dataset == 'cifar' and args.test_only:
    # for cifar only
    args.phase = 'test'
    args.max_epoch = 0

if not os.path.exists(args.save_folder):
    mkdirs(args.save_folder)

if torch.cuda.is_available():
    args.use_cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
