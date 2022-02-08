import os
import sys
import glob
import json
import click
import torch
import numpy as np

from tqdm import tqdm
from pycocotools import mask
from PIL import Image
from PIL import ImageDraw
from torch import nn

sys.path.append("../helpers")
from post_processing import remove_connected_components

sys.path.append("/home/amvernet/dhSegment-torch/")
from dh_segment_torch.data.annotation import AnnotationPainter, AnnotationReader, VIA2Reader
from dh_segment_torch.data import ColorLabels


@click.command()
@click.option('--predictions-path', '-p',
              help="""Path where the predictions of a run has been saved.""")
@click.option('--score-threshold', '-s', default=0.05,
              help="""Remove all object instance predictions with a confidence
              score below this threshold.
              Default: 0.05.""")
@click.option('--bad-predictions-threshold', '-b', default=0.05,
              help="""Output all predictions with mIoU smaller than the given
              threshold.
              Default: 0.05.""")
@click.option('--good-predictions-threshold', '-g', default=0.95,
              help="""Output all predictions with mIoU larger than the given
              threshold.
              Default: 0.95.""")
@click.option('--cc-threshold', '-t', default=0.005,
              help="""Connected components of size smaller than the
              threshold*the image size are discarded.
              Default: 0.005.""")
@click.option('--images-path', '-i',
              default="/scratch/students/amvernet/datasets/images/",
              help="""Path to the folder containing the images to be used.
              Default: /scratch/students/amvernet/datasets/images/""")
def compute_results(predictions_path, score_threshold, bad_predictions_threshold, good_predictions_threshold, cc_threshold, images_path):

    predictions_path = os.path.abspath(predictions_path)
    dataset_name = os.path.basename(predictions_path)
    run_name = os.path.basename(os.path.dirname(predictions_path))
    experiment_path = os.path.abspath(os.path.join(predictions_path, "../../.."))
    experiment_name = os.path.basename(experiment_path)

    color_labels = ColorLabels.from_labels_json_file(os.path.join(experiment_path, "color_labels.json"))
    labels = sorted(color_labels.labels)
    colors = [color_labels.colors[color_labels.labels.index(label)] for label in labels]

    annotations_path = os.path.join(experiment_path, f"via_annotations_{dataset_name}.json")
    with (open(annotations_path, "r")) as f:
        annotations = json.loads(f.read())
    predictions_path = os.path.join(predictions_path, "predictions.segm.json")
    with (open(predictions_path, "rb")) as f:
        predictions = json.load(f)
    predictions = [x for x in predictions if x['score'] >= score_threshold]

    print(f"The predictions stored in {os.path.abspath(predictions_path)} will be used in association with {annotations_path}.")
    results = {}
    print("Loading predictions...")
    for prediction in tqdm(predictions):
        k = prediction['image_id']
        if k not in results:
                results[k] = {}

        category_id = prediction['category_id'] + 1
        if "pred" not in results[k]:
            prediction_mask = mask.decode(prediction['segmentation'])
            results[k]["pred"] = np.zeros((len(labels), prediction_mask.shape[0], prediction_mask.shape[1]), dtype=np.uint8)
            results[k]["pred"][category_id] = prediction_mask
        else:
            results[k]["pred"][category_id] = (results[k]["pred"][category_id] | mask.decode(prediction['segmentation']))

    print("Loading ground-truths...")
    for k, v in tqdm(annotations.items()):
        image = Image.open(os.path.join(images_path, k + ".png"))
        original_shape = image.width, image.height
        ground_truth = np.zeros((len(labels), original_shape[1], original_shape[0]), dtype=np.uint8)

        for region in v["regions"]:
            shape = region['shape_attributes']
            if shape['name'] == 'rect':
                x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
                label = labels.index(region['region_attributes']['label'])
                ground_truth[label, y:y+h, x:x+w] = 1
            elif shape['name'] == 'polygon':
                color = labels.index(region['region_attributes']['label'])
                img = Image.new('L', original_shape, 0)
                polygon_coordinates = [(x, y) for x, y in zip(shape['all_points_x'], shape['all_points_y'])]
                ImageDraw.Draw(img).polygon(polygon_coordinates, outline=1, fill=1)
                m = np.array(img)
                np.putmask(ground_truth[label], m, 1)
            else:
                raise NotImplementedError

        if k not in results:
            results[k] = {}

        results[k]["gt"] = ground_truth

        if 'pred' not in results[k]:
            results[k]["pred"] = np.zeros(ground_truth.shape)

    print("The IoU of every predictions is being computed...")
    for pid, v in tqdm(list(results.items())):
        gt = np.argmax(v['gt'], axis=0)
        gt = np.expand_dims(gt, axis=2)
        m = nn.Upsample(size=gt.shape[:-1], mode='nearest')

        if 'pred' in v:
            pred = torch.from_numpy(v['pred']).unsqueeze(0)
            pred = np.array(m(pred).squeeze())
            pred = np.argmax(pred, axis=0)# because we only have one class here (no background), when we squeeze it removes dim 0 and 1 because they're (1,1,w,h)
            pred = np.expand_dims(pred, axis=2)
            pred = remove_connected_components(pred, cc_threshold)
        else:
            pred = np.zeros(gt.shape, dtype=np.uint8)


        results[pid]['iou'] = {}
        for i, label in list(enumerate(labels))[1:]:

            pred_i = pred.copy()
            pred_i[pred_i != i] = 0

            gt_i = gt.copy()
            gt_i[gt_i != i] = 0

            intersection = np.unique(pred_i & gt_i, return_counts=True)
            intersection = {i: val for i, val in zip(intersection[0], intersection[1])}
            union = np.unique(pred_i | gt_i, return_counts=True)
            union = {i: val for i, val in zip(union[0], union[1])}

            if i not in union:
                iou = None
            elif i not in intersection:
                iou = 0
            else:
                iou = intersection[i]/union[i]

            results[pid]['iou'][label] = iou

        miou = [v for k, v in results[pid]['iou'].items() if v is not None]
        results[pid]['miou'] = np.mean(miou) if len(miou) > 0 else None

    results_path = os.path.join(experiment_path, "mmDetection_results", run_name, dataset_name)
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, f"IoU_{score_threshold}.json"), "w") as f:
        results_dump = {k: {'iou': v['iou'], 'miou': v['miou']} for k, v in results.items()}
        json.dump(results_dump, f)

    if bad_predictions_threshold > 0 or good_predictions_threshold < 1:
        worst_predictions_path = os.path.join(results_path, f"worst_predictions_{score_threshold}")
        os.makedirs(worst_predictions_path, exist_ok=True)

        best_predictions_path = os.path.join(results_path, f"best_predictions_{score_threshold}")
        os.makedirs(best_predictions_path, exist_ok=True)

        print(f"Predictions with a mIoU smaller than {bad_predictions_threshold} or larger than {good_predictions_threshold} are being constructed and output...")

        results_filtered = {k: v for k, v in results.items() if v['miou'] is not None}
        results_filtered = {k: v for k, v in results_filtered.items() if v['miou'] < bad_predictions_threshold or v['miou'] > good_predictions_threshold }

        for pid, result in tqdm(results_filtered.items()):
            miou = result['miou']

            image = Image.open(os.path.join(images_path, pid + ".png"))
            shape = (result['gt'].shape[2], result['gt'].shape[1])
            image = image.resize(shape)
            image = np.asarray(image)

            m = nn.Upsample(size=result['gt'].shape[1:], mode='nearest')
            pred = torch.from_numpy(result['pred']).unsqueeze(0)
            pred = np.array(m(pred).squeeze())

            prediction = image//2
            ground_truth = image//2
            for i, color in list(enumerate(colors))[1:]:
                pred_i = np.expand_dims(pred[i], axis=2)
                pred_i = np.round(pred_i)
                pred_i = remove_connected_components(pred_i, cc_threshold)
                pred_i = (pred_i[:,:,:]*color).astype(np.uint8)
                prediction += pred_i//2

                gt_i = np.expand_dims(result['gt'][i], axis=2)
                gt_i = (gt_i[:,:,:]*color).astype(np.uint8)
                ground_truth += gt_i//2

            save_path = best_predictions_path if miou > good_predictions_threshold else worst_predictions_path
            Image.fromarray(prediction).save(os.path.join(save_path, f'{pid}_pred_{miou:0.3f}.png'))
            Image.fromarray(ground_truth).save(os.path.join(save_path, f'{pid}_gt.png'))


if __name__ == '__main__':
    compute_results()
