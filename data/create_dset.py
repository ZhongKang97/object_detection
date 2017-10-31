import os
from data.augmentations import SSDAugmentation
from data import AnnotationTransform
import torchvision.datasets as dset
import torchvision.transforms as transforms


def create_dataset(opts):
    means = (104, 117, 123)
    name = opts.dataset
    home = os.path.expanduser("~")
    if name == 'voc':
        print('Loading Dataset...')
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        data_root = os.path.join(home, "data/VOCdevkit/")
        from data import VOCDetection
        dataset = VOCDetection(data_root, train_sets,
                               SSDAugmentation(opts.ssd_dim, means),
                               AnnotationTransform())
    elif name == 'coco':
        data_root = os.path.join(home, 'dataset/coco')

        from data import COCODetection
        dataset = COCODetection(root=data_root, phase=opts.phase,
                                transform=SSDAugmentation(opts.ssd_dim, means))
        # dataset = dset.CocoDetection(root=(data_root + '/train2014'),
        #                              annFile=(data_root + '/annotations/' + anno_file),
        #                              transform=transforms.ToTensor())
    print('Training SSD on', dataset.name)

    return dataset
