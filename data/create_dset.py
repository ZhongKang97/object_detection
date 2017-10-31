import os
from data.augmentations import SSDAugmentation
from data import AnnotationTransform, BaseTransform
import torchvision.datasets as dset
import torchvision.transforms as transforms


def create_dataset(opts):
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
    print('{:s} SSD on {:s}'.format(opts.phase, dataset.name))

    return dataset
