from __future__ import print_function
from torch.autograd import Variable
from data.create_dset import create_dataset
from ssd import build_ssd
from option.test_opt import args
from utils.util import *
from utils.eval import *

# write options and result
args.log_file_name = os.path.join(args.save_folder,
                                  'opt_{:s}_result.txt'.format(args.phase))
with open(args.log_file_name, 'wt') as log_file:
    log_file.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        log_file.write('%s: %s\n' % (str(k), str(v)))
    log_file.write('-------------- End ----------------\n\n')
    log_file.write('================ Result ================\n')


dataset = create_dataset(args)
net, _ = build_ssd(args, dataset.num_classes)

_temp = '' if args.sub_folder_suffix == '' else '_'  # for display in a neat manner
show_freq = args.show_freq
_t = {'im_detect': Timer(), 'misc': Timer()}
num_images = len(dataset)

det_file = os.path.join(args.save_folder, 'detections_all_boxes.pkl')
if os.path.isfile(det_file):
    print('Raw boxes exist! skip test time and directly evaluate!...')
    # all_boxes = []
    # with open(det_file, 'rb') as f:
    #     while True:
    #         try:
    #             all_boxes.append(pickle.load(f))
    #         except EOFError:
    #             break
    all_boxes = pickle.load(open(det_file, 'rb'))
else:
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(dataset.num_classes)]

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
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
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        if i % show_freq == 0:
            print('[{:s}][{:s}]\tim_detect: '
                  '{:d}/{:d} {:.3f}s'.format(args.experiment_name,
                                             (os.path.basename(args.trained_model) + _temp + args.sub_folder_suffix),
                                             i + 1, num_images, detect_time))
    # save the raw boxes
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

print('Evaluating detections')
if dataset.name == 'COCO':
    res_file = det_file[:-3] + 'json'
    write_coco_results_file(dataset, all_boxes, res_file)
    coco_do_detection_eval(dataset, res_file, args.save_folder)
else:
    write_voc_results_file(args.save_folder, all_boxes, dataset)
    do_python_eval(args)
