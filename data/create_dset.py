import os
from data.augmentations import SSDAugmentation
from data import AnnotationTransform, BaseTransform
import torchvision.datasets as dset
import torchvision.transforms as T
import torch


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
        # add the data augmentation here
        if phase == 'train':
            if opts.multi_crop_test:
                transform = T.Compose([
                    T.Resize(40),
                    T.RandomCrop(32),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
                    T.ToTensor(),
                ])
            else:
                transform = T.Compose([
                    T.Resize(40),
                    T.RandomCrop(32),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            if opts.multi_crop_test:
                transform = T.Compose([
                    T.Resize(40),
                    T.TenCrop(32),
                    T.Lambda(lambda crops:
                             torch.stack([T.ToTensor()(crop) for crop in crops]))  # returns a 4D tensor
                ])
            else:
                transform = T.Compose([
                    T.Resize(40),
                    T.RandomCrop(32),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        dataset = dset.CIFAR10(root='data', train=phase == 'train',
                               transform=transform, download=True)
        dataset.name = 'CIFAR10'
        dataset.num_classes = 10

    show_phase = opts.phase if phase is None else phase
    print('{:s} on {:s}'.format(show_phase, dataset.name))

    return dataset
