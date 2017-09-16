# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
VOCroot = os.path.join(home, "data/VOCdevkit/")

# SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
# for v1, check the original repo
v2 = {
    'feature_maps':     {
        '300':      [38, 19, 10, 5, 3, 1],
        '512':      [64, 32, 16, 8, 6, 4],
    },
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.9,
    'beyond_max':       1.0,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2',
}

v3 = {
    'feature_maps':     {
        '300':      [38, 19, 10, 5, 3, 1],
        '512':      [64, 32, 16, 8, 6, 4],
    },
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.8,
    'beyond_max':       1.05,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v3',
}

# damn it
v2_512 = {
    'image_size':       512,
    'steps':            [8, 16, 32, 64, 86, 128],
    'feature_maps':     [64, 32, 16, 8, 6, 4],
    'min_sizes':        [45, 100, 150, 256, 360, 460],
    'max_sizes':        [100, 150, 256, 360, 460, 530],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2_512',
}

# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [4, 6, 6, 6, 4, 4],
# }
