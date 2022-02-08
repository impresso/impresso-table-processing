import os
from pathlib import Path

import boto3
import pysolr
import pandas as pd


# model used for the GUI to show similar images
SIMILARITY_MODEL_NAME = "ResNet50"

# parameters for auto-tagging
AUTO_TAG_CONFIG = {"score_threshold": 0.99,
                   "model_names": ["ResNet50", "VGG16", "InceptionResNetV2"],
                   "same_journal": True,
                   "timespan": 365*5}


### SOLR (not mandatory if visual similarity is not used)
# where the image vectors for tables have been stored
solr = pysolr.Solr(os.environ.get('SOLR_URL', None) + "/impresso_table_images")


### S3 (not mandatory if dataset to be used is locally stored)
IMAGES_ON_S3 = True  # are the images stored locally or on S3
# where images are stored on S3
s3 = boto3.resource('s3',
                    endpoint_url='https://' + os.environ.get("S3_ENDPOINT", 'os.zhdk.cloud.switch.ch'),
                    aws_access_key_id=os.environ.get("SE_ACCESS_KEY", None),
                    aws_secret_access_key=os.environ.get("SE_SECRET_KEY", None))
bucket = s3.Bucket("impresso-tables")


### CACHE FILE
# create a file locally that stores images browsed
MAX_CACHE_SIZE = 1000000000  # 1GB
# where the cache will be stored
TMP_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
os.makedirs(TMP_PATH, exist_ok=True)


### DATASET
# where anything related to the dataset should be saved
DATASET_NAME = "NLL"
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", DATASET_NAME)
os.makedirs(DATASET_PATH, exist_ok=True)
# path to the parquet file that contains the location of the cropped and full images
METADATA_PATH = os.path.join(DATASET_PATH, "tables_metadata_s3.parquet")
if Path(METADATA_PATH).exists():
    DF_METADATA = pd.read_parquet(METADATA_PATH)
else:
    raise ValueError
# path to where the tags will be saved
TAGS_PATH = os.path.join(DATASET_PATH, "tags.parquet")
if Path(TAGS_PATH).exists():
    DF_TAGS = pd.read_parquet(TAGS_PATH)
else:
    DF_TAGS = pd.DataFrame(data=DF_METADATA["id"], columns=["id", "tag", "date_tag"])
    DF_TAGS["tag"] = ""
# path where the auto-tags will be saved
AUTO_TAGS_PATH = os.path.join(DATASET_PATH, "auto_tags.parquet")
# set to True to load the dataset created by the auto-tagging instead
AUTO = False
if AUTO:
    df_auto_tags = pd.read_json(os.path.join(DATASET_PATH, "auto_tags.json"), orient='index')\
        .reset_index().rename({"index": "id"}, axis=1)[['id', 'tag', 'date_tag', 'auto_tag']]
    df_auto_tags['date_tag'] = pd.to_datetime(df_auto_tags['date_tag']).dt.tz_localize(None)
    DF_TAGS = pd.merge(DF_TAGS['id'], df_auto_tags, how='left', on='id')
    DF_TAGS['tag'] = DF_TAGS['tag'].fillna('')
