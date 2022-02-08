import pickle
import sys
import gzip
import json

from helpers import is_page
from settings import *


images = dict()
current_cache_size = 0
CACHE_PATH = os.path.join(TMP_PATH, "images_bytes_b64.pkl")
if Path(CACHE_PATH).exists():
    with open(CACHE_PATH, "rb") as f:
        images = pickle.load(f)
    current_cache_size = os.path.getsize(CACHE_PATH)


def fetch_image(image_id):
    if image_id not in images:
        if is_page(image_id):
            obj = bucket.Object(DF_METADATA.loc[DF_METADATA['pid'] == image_id, 'pid_loc'].iloc[0])
        else:
            obj = bucket.Object(DF_METADATA.loc[DF_METADATA['id'] == image_id, 'id_loc'].iloc[0])

        with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
            for line in gzipfile.readlines():
                line = json.loads(line)
                if line['id'] == image_id:
                    requested_image = line['bytes_b64']
                    break
                elif not is_page(image_id):
                    images[line['id']] = line['bytes_b64']
    else:
        requested_image = images[image_id]
        del images[image_id]

    images[image_id] = requested_image


def trim_cache(max_size_table=1 / 3.0, max_size_page=2 / 3.0, num_tables_to_delete=30, num_images_to_delete=10):
    full_images_size = sum([sys.getsizeof(v) for k, v in images.items() if is_page(k)])
    while full_images_size > max_size_page * MAX_CACHE_SIZE:
        for _ in range(num_images_to_delete):
            del images[list(images.keys())[0]]
        full_images_size = sum([sys.getsizeof(v) for k, v in images.items() if is_page(k)])

    table_images_size = sum([sys.getsizeof(v) for k, v in images.items() if not is_page(k)])
    while table_images_size > max_size_table * MAX_CACHE_SIZE:
        for _ in range(num_tables_to_delete):
            del images[list(images.keys())[0]]
        table_images_size = sum([sys.getsizeof(v) for k, v in images.items() if not is_page(k)])


def save_cache(size_interval=1000000):
    global current_cache_size
    images_size = sum([sys.getsizeof(v) for k, v in images.items()])
    if images_size - current_cache_size > size_interval:
        with open(CACHE_PATH, "wb") as cache_file:
            pickle.dump(images, cache_file)
        current_cache_size = os.path.getsize(CACHE_PATH)
