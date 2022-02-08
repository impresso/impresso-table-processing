import json
import numpy as np


def miou(ious, label='miou'):
    if label == 'miou':
        return np.mean([v['miou'] for k, v in ious.items() if v['miou'] is not None])
    else:
        return np.mean([v['iou'][label] for k, v in ious.items() if v['iou'][label] is not None])


def precision_at(ious, threshold, label='miou'):
    threshold = threshold*0.01
    if label == 'miou':
        tp = {k: v for k, v in ious.items() if v['miou'] is not None and v['miou'] >= threshold}
        tn = {k: v for k, v in ious.items() if v['miou'] is None}
        fn = {k: v for k, v in ious.items() if v['miou'] is not None and v['miou'] == 0.0}
        fp = {k: v for k, v in ious.items() if v['miou'] is not None and v['miou'] < threshold and k not in fn}
    else:
        tp = {k: v for k, v in ious.items() if v['iou'][label] is not None and v['iou'][label] >= threshold}
        tn = {k: v for k, v in ious.items() if v['iou'][label] is None}
        fn = {k: v for k, v in ious.items() if v['iou'][label] is not None and v['iou'][label] == 0.0}
        fp = {k: v for k, v in ious.items() if v['iou'][label] is not None and v['iou'][label] < threshold and k not in fn}

    if (len(tp)+len(fp)) == 0:
        return 0

    return len(tp)/(len(tp)+len(fp))


def recall_at(ious, threshold, label='miou'):
    threshold = threshold*0.01
    if label == 'miou':
        tp = {k: v for k, v in ious.items() if v['miou'] is not None and v['miou'] >= threshold}
        fn = {k: v for k, v in ious.items() if v['miou'] is not None and v['miou'] == 0.0}
    else:
        tp = {k: v for k, v in ious.items() if v['iou'][label] is not None and v['iou'][label] >= threshold}
        fn = {k: v for k, v in ious.items() if v['iou'][label] is not None and v['iou'][label] == 0.0}

    if (len(tp)+len(fn)) == 0:
        return 0

    return len(tp)/(len(tp)+len(fn))


def average_precision(ious, start_threshold, end_threshold, step, label='miou'):
    precisions = []
    thresholds = range(start_threshold, end_threshold + step, step)
    for threshold in thresholds:
        precision = precision_at(ious, threshold, label)
        precisions.append(precision)

    return np.mean(precisions)


def average_recall(ious, start_threshold, end_threshold, step, label='miou'):
    recalls = []
    thresholds = range(start_threshold, end_threshold + step, step)
    for threshold in thresholds:
        recall = recall_at(ious, threshold, label)
        recalls.append(recall)

    return np.mean(recalls)


def compute_metrics(iou_paths, label='miou'):
    metrics = {"mIoU": [],
           "P@60": [],
           "R@60": [],
           "P@80": [],
           "R@80": [],
           "P@50:5:95": [],
           "R@50:5:95": []}

    for iou_path in iou_paths:
        with open(iou_path, "r") as f:
            iou = json.load(f)

        metrics["mIoU"].append(miou(iou, label))
        metrics["P@60"].append(precision_at(iou, 60, label))
        metrics["R@60"].append(recall_at(iou, 60, label))
        metrics["P@80"].append(precision_at(iou, 80, label))
        metrics["R@80"].append(recall_at(iou, 80, label))
        metrics["P@50:5:95"].append(average_precision(iou, 50, 95, 5, label))
        metrics["R@50:5:95"].append(average_recall(iou, 50, 95, 5, label))

    for k, v in metrics.items():
        metrics[k] = np.mean(v), np.std(v)


    return metrics
