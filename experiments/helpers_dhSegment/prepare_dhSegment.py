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
              help="""Path where fixed annotations, color labels and dataset
              metadata are output.""")
@click.option('--annotations-path', '-a',
              help="""Path to a VIA file containing the dataset to be used
              (must be named \"via_annotations_<dataset_name>.json\").""")
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

    if os.path.basename(annotations_path)[:16] != "via_annotations_":
        raise ValueError("Annotations file name should start with via_annotations_")
    dataset_name = os.path.basename(annotations_path)[16:-5]

    tagset = {region['region_attributes']['label'] for k, v in annotations.items() for region in v['regions']}
    if verbose:
        print(f"The tagset is the following: {tagset}.")
        print(f"Annotations are being scaled to values found in {metadata_path}.")

    for k, v in annotations.items():
        v['filename'] = os.path.join(images_path, v['filename'])
        scale_factor = metadata[k]['resized_height']/metadata[k]['height'] if scale else 1
        for region in v['regions']:
            r = region['shape_attributes']
            if r['name'] == 'rect':
                region['shape_attributes']['x'] = int(scale_factor*r['x'])
                region['shape_attributes']['y'] = int(scale_factor*r['y'])
                region['shape_attributes']['width'] = int(scale_factor*r['width'])
                region['shape_attributes']['height'] = int(scale_factor*r['height'])
            elif r['name'] == 'polygon':
                region['shape_attributes']['all_points_x'] = list(map(lambda x: x*scale_factor, r['all_points_x']))
                region['shape_attributes']['all_points_y'] = list(map(lambda x: x*scale_factor, r['all_points_y']))

    os.makedirs(join(output_path, "dhSegment_models/0"), exist_ok=True)
    annotations_filename = f"via_annotations_{dataset_name}.json"
        
    with open(os.path.join(output_path, annotations_filename), "w") as f:
        json.dump(annotations, f)

    via = VIA2Reader.from_annotation_file("label", os.path.join(output_path, annotations_filename))

    color_dict = {'background': tuple([int(x*255) for x in colors.to_rgb('black')]),
                  'exchange': tuple([int(x*255) for x in colors.to_rgb('gold')]),
                  'transport schedule': tuple([int(x*255) for x in colors.to_rgb('blue')]),
                  'food prices': tuple([int(x*255) for x in colors.to_rgb('green')]),
                  'sport results': tuple([int(x*255) for x in colors.to_rgb('lime')]),
                  'table': tuple([int(x*255) for x in colors.to_rgb('lightsteelblue')]),
                  'miscellaneous': tuple([int(x*255) for x in colors.to_rgb('lightsteelblue')]),
                  'weather': tuple([int(x*255) for x in colors.to_rgb('cyan')]),
                  'not a table': tuple([int(x*255) for x in colors.to_rgb('magenta')]),
                  'radio': tuple([int(x*255) for x in colors.to_rgb('darkviolet')]),
                  'election': tuple([int(x*255) for x in colors.to_rgb('lightcoral')]),
                  'lotto': tuple([int(x*255) for x in colors.to_rgb('olive')]),
                  'cinema': tuple([int(x*255) for x in colors.to_rgb('red')])}
    labels = ['background'] + list(tagset)
    c = [color_dict[label] for label in labels]
    color_labels = ColorLabels(c, labels=labels)

    if verbose:
        print("The following colors and labels are used:\n{}\nIf you wish to change them (or the possible labels), you need to modify the source code directly.".format({k: v for k, v in color_dict.items() if k in tagset}))

    labels_path = os.path.join(os.path.dirname(annotations_path), "../dhSegment_labels")
    labels_dir = os.path.abspath(os.path.join(labels_path, f"{dataset_name}"))

    if verbose:
        print(f"Labels will be saved under {labels_dir}.")

    img_writer = ImageWriter.from_reader(annotation_reader=via,
                                         color_labels=color_labels,
                                         labels_dir=labels_dir,
                                         csv_path=os.path.join(output_path, f"{dataset_name}.csv"),
                                         color_labels_file_path=os.path.join(output_path, "color_labels.json"),
                                         copy_images=False,
                                         overwrite=overwrite,
                                         progress=verbose)

    if verbose:
        print("Labels are being generated...")

    image_to_label = img_writer.write(num_workers=32)


if __name__ == '__main__':
    prepare_experiment()
