import os
from utils.augmentations import SSDAugmentation
from data import VOCDetection, AnnotationTransform
import torchvision.datasets as dset
import torchvision.transforms as transforms


def create_dataset(opts):
    name = opts.dataset
    home = os.path.expanduser("~")
    if name == 'voc':
        print('Loading Dataset...')
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        data_root = os.path.join(home, "data/VOCdevkit/")
        means = (104, 117, 123)
        dataset = VOCDetection(data_root, train_sets,
                               SSDAugmentation(opts.ssd_dim, means), AnnotationTransform())
    elif name == 'coco':
        data_root = os.path.join(home, 'dataset/coco')
        anno_file = 'instances_train2014.json'
        dataset = dset.CocoDetection(root=(data_root + '/train2014'),
                                     annFile=(data_root + '/annotations/' + anno_file),
                                     transform=transforms.ToTensor())
    print('Training SSD on', dataset.name)

    return dataset
