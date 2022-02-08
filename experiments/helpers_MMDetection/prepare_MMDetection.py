import os
import sys
import json
import click
import pandas as pd

from PIL import Image
from os import listdir
from os.path import isfile, join
from matplotlib import colors

sys.path.append("/home/amvernet/dhSegment-torch/")
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.data.annotation import VIA2Reader, ImageWriter, ColorLabels
from dh_segment_torch.data.transforms import FixedResize


@click.command()
@click.option('--output-path', '-o',
              help="""Path where fixed annotations are output.""")
@click.option('--annotations-path', '-a',
              help="""Path to a COCO file containing the dataset to be used
              (must be named \"coco_annotations_<dataset_name>.json\").""")
@click.option('--overwrite', default=False, is_flag=True,
              help="""Whether or not to overwrite already existing label images.
              Default: False.""")
@click.option('--scale', default=True, is_flag=True,
              help="""Whether or not to scale the annotations to the image size.
              Default: True.""")
@click.option('--images-path', '-i',
              default="/scratch/students/amvernet/datasets/images/",
              help="""Path to the folder containing the images to be used.
              Default: /scratch/students/amvernet/datasets/images/""")
@click.option('--metadata-path', '-m',
              default="/scratch/students/amvernet/datasets/metadata/",
              help="""Path to folder with JSON files containing the original and resized
              height and width of the pictures to be used.\n
              Default: /scratch/students/amvernet/datasets/metadata""")
@click.option('--verbose', '-v', default=True, is_flag=True)
def prepare_experiment(output_path, annotations_path, overwrite, scale, images_path, metadata_path, verbose):

    if verbose:
        print(f"Data from {os.path.abspath(annotations_path)} about to be pre-processed and transfered to {os.path.abspath(output_path)}.")

    with (open(annotations_path, "r")) as f:
        annotations = json.loads(f.read())

    metadata = {}
    onlyfiles = [f for f in listdir(metadata_path) if isfile(join(metadata_path, f))]
    for file in onlyfiles:
        if file.endswith(".json"):
            with (open(join(metadata_path, file), "r")) as f:
                metadata.update(json.loads(f.read()))

    if os.path.basename(annotations_path)[:17] != "coco_annotations_":
        raise ValueError("Annotations file name should start with coco_annotations_")
    dataset_name = os.path.basename(annotations_path)[17:-5]

    tagset = set(x['name'] for x in annotations['categories'])
    if verbose:
        print(f"The tagset is the following: {tagset}.")
        print(f"Annotations are being scaled to values found in {metadata_path}.")

    for image in annotations['images']:
        pid = image['id']
        image['file_name'] = os.path.join(images_path, image['file_name'])
        if scale:
            image['height'] = metadata[pid]['resized_height']
            image['width'] = metadata[pid]['resized_width']

    if scale:
        for annotation in annotations['annotations']:
            pid = annotation['image_id']
            scale_factor = metadata[pid]['resized_height']/metadata[pid]['height'] if scale else 1
            for i, mask in enumerate(annotation['segmentation']):
                annotation['segmentation'][i] = [int(scale_factor*p) for p in mask]
            annotation['area'] = int(scale_factor*annotation['area'])
            annotation['bbox'] = [int(scale_factor*p) for p in annotation['bbox']]

    os.makedirs(join(output_path, "mmDetection_models/0"), exist_ok=True)
    annotations_filename = f"coco_annotations_{dataset_name}.json"

    with open(os.path.join(output_path, annotations_filename), "w") as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    prepare_experiment()
