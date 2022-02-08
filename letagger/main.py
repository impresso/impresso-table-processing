import base64

import pandas as pd

from cache import *
from helpers import *
from settings import *
from datetime import datetime

from flask import Flask, render_template, request

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def main():
    df_no_tags = DF_TAGS[DF_TAGS['tag'] == '']
    df_no_tags['issue'] = df_no_tags['id'].apply(get_meta_issue_id)
    weights = get_weights_across_dimension(df_no_tags, 'issue') if len(df_no_tags) > 0 else None
    dataset = df_no_tags.sample(min(500, df_no_tags.shape[0]), random_state=0, weights=weights).sort_index()

    existing_tags = list(DF_TAGS['tag'].unique())
    if '' in existing_tags:
        existing_tags.remove('')
    data = {'tags': existing_tags}

    return render_template('index.html',
                           id_tags=[(row['id'], row['tag']) for i, row in dataset[['id', 'tag']].iterrows()],
                           data=data)


@app.route("/filter", methods=['GET'])
def filter_items():
    tag_type = request.args.get('type')
    if tag_type == "tagged":
        dataset = DF_TAGS[DF_TAGS['tag'] != ''].sort_values(by='date_tag', ascending=False)
    elif tag_type == "untagged":
        df_no_tags = DF_TAGS[DF_TAGS['tag'] == '']
        df_no_tags['issue'] = df_no_tags['id'].apply(get_meta_issue_id)
        weights = get_weights_across_dimension(df_no_tags, 'issue')
        dataset = df_no_tags.sample(min(500, df_no_tags.shape[0]), random_state=0, weights=weights).sort_index().sample(
            1000)
    elif tag_type == "to be tagged":
        ids_tagged = DF_TAGS.loc[DF_TAGS['tag'] != '', 'id']
        page_ids = DF_METADATA.loc[DF_METADATA['id'].isin(ids_tagged), 'pid']
        ids_to_be_tagged = DF_METADATA.loc[DF_METADATA['pid'].isin(page_ids), 'id']
        dataset = DF_TAGS[(DF_TAGS['id'].isin(ids_to_be_tagged)) & (DF_TAGS['tag'] == '')].sort_values('id')
    elif tag_type == "auto-tagged":
        dataset = DF_TAGS[DF_TAGS['auto_tag']].sample(500).sort_values('tag')
    else:
        dataset = DF_TAGS[DF_TAGS['tag'] == tag_type].sort_values(by='id')

    response = json.dumps([{"id": row['id'], "tag": row['tag']} for i, row in dataset[['id', 'tag']].iterrows()])

    return response


@app.route("/img", methods=['GET'])
def get_image():
    requested_id = request.args.get('id')
    if IMAGES_ON_S3:
        fetch_image(requested_id)
        trim_cache()
        save_cache()
        return images[requested_id]
    else:
        with open(DF_METADATA.loc[DF_METADATA["id"] == requested_id, "id_loc"].iloc[0], "rb") as image:
            return base64.b64encode(image.read())


@app.route("/zoom", methods=['GET'])
def zoom():
    requested_id = DF_METADATA.loc[DF_METADATA['id'] == request.args.get('id'), 'pid'].iloc[0]

    if IMAGES_ON_S3:
        fetch_image(requested_id)
        trim_cache()
        save_cache()
        return images[requested_id]
    else:
        with open(DF_METADATA.loc[DF_METADATA["pid"] == requested_id, "pid_loc"].iloc[0], "rb") as image:
            return base64.b64encode(image.read())


@app.route("/sim", methods=['GET'])
def similar():
    requested_id = request.args.get('id')
    field_name = f"_vector_{SIMILARITY_MODEL_NAME}_bv"
    params = {'rows': 26}  # we want the top 25 results, and one of the 26 will be the one we want to compare against
    input_vector_b64 = list(solr.search(f"id:{requested_id}", fl=field_name))[0][field_name]
    similar_images_list = list(solr.search(f'{{!vectorscoring f="{field_name}" vector_b64="{input_vector_b64}"}}',
                                           fl="id, score",
                                           fq="", **params))

    similar_images_list = [x for x in similar_images_list if x['id'] != requested_id]
    tags = DF_TAGS.loc[DF_TAGS['id'].isin([x['id'] for x in similar_images_list]), ['id', 'tag']]
    for x in similar_images_list:
        x.update({"tag": tags.loc[tags['id'] == x['id'], 'tag'].iloc[0]})

    fetch_image(requested_id)

    response = json.dumps(similar_images_list)

    return response


@app.route("/save", methods=['POST'])
def save_tags():
    tags = request.get_json()
    for k, v in tags.items():
        DF_TAGS.loc[DF_TAGS['id'] == k, 'tag'] = v['tag']
        DF_TAGS.loc[DF_TAGS['id'] == k, 'date_tag'] = v['date']
        DF_TAGS.loc[DF_TAGS['id'] == k, 'auto_tag'] = False

    if not AUTO:
        DF_TAGS.to_parquet(TAGS_PATH)
    else:
        DF_TAGS.to_parquet(AUTO_TAGS_PATH)

    return "The tags have been saved."


@app.route("/extract", methods=['GET'])
def extract():
    is_tags = request.args.get('tags') == 'true'
    tags = DF_TAGS.loc[DF_TAGS['tag'] != '', ['id', 'tag', 'date_tag', 'auto_tag']].set_index('id').to_json(
        orient='index', date_format='iso')
    if is_tags:
        extract_path = os.path.join(DATASET_PATH, f"tags_{datetime.now().replace(microsecond=0)}.json")
        with open(extract_path, "w") as f:
            f.write(tags)
    else:
        extract_path = os.path.join(DATASET_PATH, f"dataset_{datetime.now().replace(microsecond=0)}")
        os.makedirs(extract_path, exist_ok=True)
        with open(os.path.join(DATASET_PATH, "tags.json"), "w") as f:
            f.write(tags)

        for i, row in DF_TAGS[DF_TAGS['tag'] != ''].iterrows():
            if IMAGES_ON_S3:
                fetch_image(row['id'])
                image = images[row['id']]
                image = base64.b64decode(image)
            else:
                with open(DF_METADATA.loc[DF_METADATA["id"] == row['id'], "pid_loc"].iloc[0], "rb") as image:
                    image = image.read()
            tag_path = os.path.join(extract_path, row['tag'])
            os.makedirs(tag_path, exist_ok=True)
            with open(os.path.join(tag_path, DF_METADATA.loc[DF_METADATA["id"] == row['id'], "pid"].iloc[0] + ".png"),
                      'wb') as f:
                f.write(image)

    return f"Finished extracting tags{'.' if is_tags else 'and images.'} The data can be found at {extract_path}."


@app.route("/auto", methods=['GET'])
def auto_tag():
    same_journal = AUTO_TAG_CONFIG['same_journal']
    timespan = AUTO_TAG_CONFIG['timespan']

    forbidden_ids = set()
    df_tags_copy = DF_TAGS.copy()
    for _, row in DF_TAGS[(DF_TAGS['tag'] != '') & (~DF_TAGS['auto_tag'])].iterrows():
        forbidden_ids.add(row['id'])
        similar_images_across_models = set()
        for i, model_name in enumerate(AUTO_TAG_CONFIG['model_names']):
            field_name = f"_vector_{model_name}_bv"
            input_vector_b64 = list(solr.search(f"id:{row['id']}", fl=field_name))[0][field_name]
            results = solr.search(f'{{!vectorscoring f="{field_name}" vector_b64="{input_vector_b64}"}}',
                                  fl="id, score",
                                  rows=250)

            similar_images = {result['id'] for result in results
                              if result['score'] > AUTO_TAG_CONFIG['score_threshold']
                              and result['id'] not in forbidden_ids}

            if same_journal:
                similar_images = {k for k in similar_images
                                  if get_journal(k) == get_journal(row['id'])}

            if timespan > 0:
                similar_images = {k for k in similar_images
                                  if (get_date(k) - get_date(row['id'])).days <= timespan}

            if i == 0:
                similar_images_across_models = similar_images
            else:
                similar_images_across_models = similar_images_across_models.intersection(similar_images)

        for k in list(similar_images_across_models):
            similar_image_manual_tag = DF_TAGS.loc[DF_TAGS['id'] == k, 'tag'].iloc[0]
            if similar_image_manual_tag not in {'', row['tag']}:
                print(f"""{k} was about to be tagged as a {row['tag']} just like {row['id']}
                 but was already tagged manually as {similar_image_manual_tag}.""")

            if similar_image_manual_tag == row['tag']:
                similar_images_across_models.remove(k)

            similar_image_auto = df_tags_copy.loc[df_tags_copy['id'] == k].iloc[0]
            similar_image_auto_tag = similar_image_auto['tag']
            if similar_image_auto_tag not in {'', row['tag']}:
                print(f"""{k} was about to be tagged as a {row['tag']} just like {row['id']}, but was already 
                tagged automatically as {similar_image_auto_tag} just like {similar_image_auto['og_id']}. 
                This image will therefore not be considered anymore.""")
                df_tags_copy.loc[df_tags_copy['id'] == k, ['auto_tag', 'same_journal', 'timespan', 'og_id', 'tag']] = ''
                forbidden_ids.add(k)
                similar_images_across_models.remove(k)

        current_time = datetime.now()
        new_row = row['tag'], current_time, True, same_journal, timespan, row['id']
        df_tags_copy.loc[df_tags_copy['id'].isin(similar_images_across_models),
                         ['tag', 'date_tag', 'auto_tag', 'same_journal', 'timespan', 'og_id']] = new_row

    save_path = os.path.join(DATASET_PATH, f"auto_tags_{datetime.now().replace(microsecond=0)}.json")
    df_tags_copy[df_tags_copy['tag'] != ''].set_index('id').to_json(save_path, orient='index', date_format='iso')

    return f"Finished auto-tagging. The .json file can be found at {save_path}."
