import pickle
from utils.util import *
from data.setup_dset import VOC_CLASSES as labelmap
import json


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    # DEPRECATED
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(save_folder, image_set, cls):

    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(save_folder, 'detection_per_cls')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    # path = result/EXPERIMENT_NAME/test/detection_per_cls/det_test_XXX_CLS.txt
    path = os.path.join(filedir, filename)
    return path


def voc_eval(detpath, annopath, imagesetfile, classname,
             cachedir, ovthresh=0.5, use_07_metric=True):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath:            Path to detections, detpath.format(classname) should produce the detection results file.
    annopath:           Path to annotations, annopath.format(imagename) should be the xml annotations file.
    imagesetfile:       Text file containing the list of images, one image per line.
    classname:          Category name (duh)
    cachedir:           Directory for caching the annotations
    [ovthresh]:         Overlap threshold (default = 0.5)
    [use_07_metric]:    Whether to use VOC07's 11 point AP computation (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def write_voc_results_file(save_folder, all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        set_type = 'test'
        filename = get_voc_results_file_template(save_folder, set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(opts):

    prefix = opts.save_folder
    log_file_name = opts.log_file_name
    save_folder = opts.save_folder
    home = os.path.expanduser("~")
    data_root = os.path.join(home, "data/VOCdevkit/")
    set_type = 'test'

    devkit_path = data_root + 'VOC' + '2007'
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
    imgsetpath = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')

    output_dir = os.path.join(prefix, 'pr_curve_per_cls')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    use_07_metric = True
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    with open(log_file_name, 'a') as log_file:
        aps = []
        for i, cls in enumerate(labelmap):
            filename = get_voc_results_file_template(save_folder, set_type, cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imgsetpath.format(set_type), cls, cachedir,
                ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            log_file.write('AP for {} = {:.4f}\n'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        log_file.write('Mean AP = {:.4f}\n'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('--------------------------------------------------------------')


def _coco_results_one_category(dataset, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(dataset.ids):
        # dets = boxes[im_ind].astype(np.float)
        dets = np.array(boxes[im_ind], dtype=np.float)
        if len(dets) == 0:
            continue
        scores = dets[:, -1]
        xs = dets[:, 0]
        ys = dets[:, 1]
        ws = dets[:, 2] - xs + 1
        hs = dets[:, 3] - ys + 1
        results.extend(
            [{'image_id': index,
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results


def write_coco_results_file(dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(dataset.COCO_CLASSES_names):
        if cls == '__background__':
            continue
        print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, dataset.num_classes - 2))
        coco_cat_id = dataset.COCO_CLASSES[cls_ind]
        results.extend(_coco_results_one_category(dataset, all_boxes[cls_ind], coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _print_detection_eval_metrics(dataset, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(dataset.COCO_CLASSES_names):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        # update: we don't have minus 1 problem
        # precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{:.1f}'.format(100 * ap))
    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


def coco_do_detection_eval(dataset, res_file, output_dir):
    ann_type = 'bbox'
    coco_dt = dataset.coco.loadRes(res_file)
    from pycocotools.cocoeval import COCOeval
    coco_eval = COCOeval(dataset.coco, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _print_detection_eval_metrics(dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'coco_det_eval_res.pkl')
    with open(eval_file, 'wb') as fid:
        pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))
