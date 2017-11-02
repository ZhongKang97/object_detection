import os
from data.augmentations import SSDAugmentation
from data import AnnotationTransform, BaseTransform
import torchvision.datasets as dset
import torchvision.transforms as T


def create_dataset(opts, phase=None):
    means = (104, 117, 123)
    name = opts.dataset
    home = os.path.expanduser("~")
    DataAug = SSDAugmentation if opts.phase == 'train' else BaseTransform

    if name == 'voc':
        print('Loading Dataset...')
        sets = [('2007', 'trainval'), ('2012', 'trainval')] if opts.phase == 'train' else [('2007', 'test')]
        data_root = os.path.join(home, "data/VOCdevkit/")
        from data import VOCDetection
        dataset = VOCDetection(data_root, sets,
                               DataAug(opts.ssd_dim, means),
                               AnnotationTransform())
    elif name == 'coco':
        data_root = os.path.join(home, 'dataset/coco')

        from data import COCODetection
        dataset = COCODetection(root=data_root, phase=opts.phase,
                                transform=DataAug(opts.ssd_dim, means))
        # dataset = dset.CocoDetection(root=(data_root + '/train2014'),
        #                              annFile=(data_root + '/annotations/' + anno_file),
        #                              transform=transforms.ToTensor())
    elif name == 'cifar':
        if phase == 'train':
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = dset.CIFAR10(root='data', train=phase == 'train',
                               transform=transform, download=True)
        dataset.name = 'CIFAR10'
        dataset.num_classes = 10
    show_phase = opts.phase if phase is None else phase
    print('{:s} SSD on {:s}'.format(show_phase, dataset.name))

    return dataset
