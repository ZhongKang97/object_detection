from __future__ import print_function
from torch.autograd import Variable

from data.create_dset import create_dataset
from .layers.ssd import build_ssd
from .option.test_opt import TestOptions
from .utils.eval import *
from .utils.visualizer import Visualizer

option = TestOptions()
option.setup_config()
args = option.opt

dataset = create_dataset(args)

# init log file
show_jot_opt(args)

# init visualizer
visual = Visualizer(args, dataset)

# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
if os.path.isfile(args.det_file):
    print_log('\nRaw boxes exist! skip prediction and directly evaluate!',
              args.file_name)
    all_boxes = pickle.load(open(args.det_file, 'rb'))
else:
    ssd_net = build_ssd(args, dataset.num_classes)
    t = Timer()
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(dataset.num_classes)]

    for i in range(num_images):

        im, _, h, w, orgin_im, im_name = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.use_cuda:
            x = x.cuda()
        t.tic()
        detections = ssd_net(x).data
        detect_time = t.toc(average=False)

        # skip j = 0 (background class)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            all_boxes[j][i] = \
                np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)

        if i % args.show_freq == 0 or i == (num_images-1):
            progress = (i, num_images, detect_time)
            show_off = (all_boxes, orgin_im, im_name)
            visual.show_image(progress, show_off)

    # save the raw boxes
    with open(args.det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

# # debug
# debug_all_boxes = [[[] for _ in range(20)] for _ in range(dataset.num_classes)]
# for i in range(20):
#     for j in range(dataset.num_classes):
#         debug_all_boxes[j][i] = all_boxes[j][i]
# all_boxes = debug_all_boxes

print_log('\nEvaluating detection results ...', args.file_name)
if dataset.name == 'COCO':
    write_coco_results_file(dataset, all_boxes, args)
    coco_do_detection_eval(dataset, args)
else:
    write_voc_results_file(args.save_folder, all_boxes, dataset)
    do_python_eval(args)

print_log('\nHurray! Testing done!', args.file_name)
