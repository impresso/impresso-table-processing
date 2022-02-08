import io
import os
import gzip
import json
import boto3
import click
import base64
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


@click.command()
@click.option('--annotations-path', '-a',
              help="""Path to a dict with keys matching the image file name
              (without file extension) to download """)
@click.option('--images-path', '-i',
              default="/scratch/students/amvernet/datasets/images/",
              help="""Path to the folder where images will be downloaded.
              Default: /scratch/students/amvernet/datasets/images/""")
@click.option('--input-path', '-i', default="/scratch/students/amvernet/data/NLL_metadata_s3.parquet",
              help="""Path to a parquet file with the S3 location to the images
              to download.""")
@click.option('--metadata-path', '-m', default="/scratch/students/amvernet/datasets/metadata/NLL_metadata_images.json",
              help="Path to a file where the original and new sizes of the
              images to be downloaded will be stored.")
@click.option('--overwrite', '-o', default=False, is_flag=True,
              help="""Whether or not to overwrite already existing images.
              Default: False.""")
@click.option('--delete', '-d', default=False, is_flag=True,
              help="""Whether or not to delete all images in images_path whose
              name is not in the file under input_path.
              Default: False.""")
@click.option('--resize', '-r', default=2e6,
              help="""Resize the images to the given area.
              Can be set to None to deactivate it.
              Default: Image total area is reduced to 2e6.""")
@click.option('--bucket-name', '-b', default="impresso-tables",
              help="""Name of the S3 bucket where to download images
              Default: impresso-tables.""")
def download_images_from_s3(annotations_path, images_path, input_path, metadata_path,
overwrite, delete, resize, bucket_name):

    if resize:
        resize = int(float(resize))

    with (open(annotations_path, "r")) as f:
        annotations = json.loads(f.read())

    if Path(metadata_path).exists():
        with (open(metadata_path, "r")) as f:
            metadata = json.loads(f.read())
    else:
        metadata = {}

    tables_s3 = pd.read_parquet(input_path)
    tables_s3 = tables_s3[tables_s3['pid'].isin(annotations.keys())].sort_values('pid_loc').drop_duplicates('pid').drop('id', axis=1)

    s3 = boto3.resource('s3',
                        endpoint_url='https://' + os.environ.get("S3_ENDPOINT", 'os.zhdk.cloud.switch.ch'),
                        aws_access_key_id=os.environ.get("SE_ACCESS_KEY", None),
                        aws_secret_access_key=os.environ.get("SE_SECRET_KEY", None))

    bucket = s3.Bucket(bucket_name)
    image_ids_to_be_downloaded = set(annotations.keys())

    if delete:
        onlyfiles = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        for file in onlyfiles:
            if os.path.splitext(file)[0] not in set(tables_s3['pid']):
                os.remove(os.path.join(images_path, file))
                print(f"Deleted: {os.path.splitext(file)[0]}")

    try:
        while len(image_ids_to_be_downloaded) > 0:
            image_id = next(iter(image_ids_to_be_downloaded))
            image_path = os.path.join(images_path, image_id + ".png")

            if (not os.path.exists(image_path) or overwrite or resize):
                zip_name = tables_s3.loc[tables_s3['pid'] == image_id, 'pid_loc'].iloc[0]
                print(f"Unzipping {zip_name}")
                obj = bucket.Object(zip_name)
                image_ids_to_be_downloaded_in_zip = set(tables_s3.loc[tables_s3['pid_loc'] == zip_name, 'pid'])

                with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
                    for line in gzipfile.readlines():

                        line = json.loads(line)

                        if line['id'] in image_ids_to_be_downloaded_in_zip:
                            save_path = os.path.join(images_path, line['id'] + ".png")

                            if (not os.path.exists(save_path) or overwrite or resize):
                                image = base64.b64decode(line['bytes_b64'])
                                image = Image.open(io.BytesIO(image))
                                height = image.height
                                width = image.width

                                if line['id'] not in metadata:
                                    metadata[line['id']] = {'height': height, 'width': width}

                                if resize:
                                    ratio = width/height
                                    new_height = int(np.sqrt(resize / ratio))
                                    new_width = int(resize / new_height)
                                    image.thumbnail((new_width, new_height))
                                    metadata[line['id']].update({'resized_height': image.height,
                                                                 'resized_width': image.width})

                                image.save(save_path)
                                image_ids_to_be_downloaded.remove(line['id'])
                                image_ids_to_be_downloaded_in_zip.remove(line['id'])
                                print(f"Saved: {line['id']}")

                        if len(image_ids_to_be_downloaded_in_zip) == 0:
                            break
            else:
                 image_ids_to_be_downloaded.remove(image_id)
    finally:
        with (open(metadata_path, "w")) as f:
            json.dump(metadata, f)
            print(f"Saved metadata under {metadata_path}.")


if __name__ == '__main__':
    download_images_from_s3()
