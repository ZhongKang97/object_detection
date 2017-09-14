# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
VOCroot = os.path.join(home, "data/VOCdevkit/")

# SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
# for v1, check the original repo
v2 = {
    # 'image_size':       300,
    # 'steps':            [8, 16, 32, 64, 100, 300],
    'feature_maps':     [38, 19, 10, 5, 3, 1],
    # 'min_sizes':        [30, 60, 111, 162, 213, 264],
    # 'max_sizes':        [60, 111, 162, 213, 264, 315],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.9,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2',
}

v2_512 = {
    # 'image_size':       512,
    # 'steps':            [8, 16, 32, 64, 86, 128],
    'feature_maps':     [64, 32, 16, 8, 6, 4],
    # 'min_sizes':        [30, 60, 111, 162, 213, 264],
    # 'max_sizes':        [60, 111, 162, 213, 264, 315],
    'aspect_ratios':    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'min_scale':        0.1,
    'max_scale':        0.9,
    'variance':         [0.1, 0.2],
    'clip':             True,
    'name':             'v2',
}

# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [4, 6, 6, 6, 4, 4],
# }
