import math
import json
import random
import numpy as np
import pandas as pd
    

def convert_df_to_coco(df, metadata_images):

    categories = set()
    images = {}
    annotations = []

    for i, row in df.iterrows():
        image_dict = dict()
        pid = row['pid']
        if pid not in images:
            image_dict['id'] = pid
            image_dict['file_name'] = pid + ".png"
            image_dict['height'] = metadata_images[pid]['height']
            image_dict['width'] =  metadata_images[pid]['width']
            images[pid] = image_dict

        annotation_dict = dict()
        annotation_dict['image_id'] = pid
        annotation_dict['id'] = len(annotations)
        annotation_dict['segmentation'] = []
        annotation_dict['iscrowd'] = 0
        annotation_dict['area'] = 0
        
        coords = row['tb_coords']
        min_x = int(coords['x'][0])
        min_y = int(coords['y'][0])
        max_w = int(coords['width'][0])
        max_h = int(coords['height'][0])
        for x, y, w, h in zip(coords['x'], coords['y'], coords['width'], coords['height']):
            x, y, w, h = int(x), int(y), int(w), int(h)
            annotation_dict['area'] += w*h
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
            annotation_dict['segmentation'].append(segmentation)
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h
            
        category = row['tag']
        annotation_dict['category_id'] = category
        annotation_dict['bbox'] = [min_x, min_y, max_w, max_h]
        annotations.append(annotation_dict)
        
        categories.add(category)

    categories = {tag: i for i, tag in enumerate(sorted(categories))}

    for x in annotations:
        x['category_id'] = categories[x['category_id']]

    categories = [{'id': v, 'name': k, 'supercategory': 'table'} for k, v in categories.items()]
    images = list(images.values())

    coco_annotations = {'images': images, 'categories': categories, 'annotations': annotations}
    
    return coco_annotations
    
    
def convert_df_to_via(df):

    via_annotations = dict()    
    for i, row in df.iterrows(): 
        pid = row['pid']
        if pid not in via_annotations:
            via_annotations[pid] = dict()
            via_annotations[pid]['filename'] = pid + '.png'
            via_annotations[pid]['size'] = -1
            via_annotations[pid]['file_attributes'] = {}
            via_annotations[pid]['regions'] = []

        coords = row['tb_coords']
        for x, y, w, h in zip(coords['x'], coords['y'], coords['width'], coords['height']):
            region = dict()
            region['shape_attributes'] = {'name': 'rect',
                                          'x': int(x),
                                          'y': int(y),
                                          'width': int(w),
                                          'height': int(h)}
            region['region_attributes'] = {'label': row['tag']}
            via_annotations[pid]['regions'].append(region)

    return via_annotations


def convert_via_to_coco(via_annotations, metadata_images):
    categories = set()
    images = []
    annotations = []
    
    for pid, v in via_annotations.items():
        image_dict = dict()
        image_dict['id'] = pid
        image_dict['file_name'] = v['filename']
        image_dict['height'] = metadata_images[pid]['height']
        image_dict['width'] =  metadata_images[pid]['width']
        images.append(image_dict)

        for region in v['regions']:
            annotation_dict = dict()
            annotation_dict['id'] = len(annotations)
            annotation_dict['image_id'] = pid
            c = region['shape_attributes']
            
            if c['name'] == 'rect':
                x, y, w, h = c['x'], c['y'], c['width'], c['height']
                annotation_dict['bbox'] = [x, y, w, h]
                annotation_dict['area'] = w*h
                annotation_dict['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                
            elif c['name'] == 'polygon':
                xs = c['all_points_x']
                ys = c['all_points_y']
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                w = max_x - min_x
                h = max_y - min_y
                segmentation = [(x,y) for x, y in zip(xs, ys)]
                annotation_dict['segmentation'] = [[item for xy_tuple in segmentation for item in xy_tuple]]
                annotation_dict['bbox'] = [x, y, w, h]
                annotation_dict['area'] = w*h
            
            annotation_dict['iscrowd'] = 0
            annotation_dict['category_id'] = region['region_attributes']['label']
            annotations.append(annotation_dict)
            categories.add(region['region_attributes']['label'])

    categories = {tag: i for i, tag in enumerate(sorted(categories))}

    for x in annotations:
        x['category_id'] = categories[x['category_id']]

    categories = [{'id': v, 'name': k, 'supercategory': 'table'} for k, v in categories.items()]

    coco_annotations = {'images': images, 'categories': categories, 'annotations': annotations}

    return coco_annotations


def stratified_split_by_id(df, splits, manual_test_tag=False, seed=0, print_output=False):
    if sum(splits) > 1:
        raise ValueError
        
    r = random.Random(seed)

    train = pd.DataFrame(columns=df.columns)
    val = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)
    
    ids = set(df.index)
    proportions_per_tag = df.groupby('tag') \
                                .count()['pid'] \
                                .apply(lambda x: x/len(ids)) \
                                .sort_values(ascending=False)
    
    test_size = int(splits[2]*len(ids))
    already_sampled_ids = set()
    print(df.shape)  
    for tag, percentage in proportions_per_tag.iteritems():  
        if manual_test_tag:
            df_tag = df.loc[~(df["auto_tag"]) & (df['tag'] == tag)]
        else:
            df_tag = df.loc[df['tag'] == tag]
        
        missing = math.ceil(test_size*percentage)
        number_to_sample = max(0, missing)
        sample = df_tag.sample(number_to_sample)
        df = df.drop(list(sample.index))
        test = pd.concat([test, sample])
    print(df.shape)    
    train_size = int(splits[0]*len(ids))
    val_size = len(ids) - train_size - test_size
    for tag, percentage in proportions_per_tag.iteritems():  
        df_tag = df.loc[df['tag'] == tag]
        if len(df_tag) > 0:
            number_to_sample = math.ceil(train_size*percentage)
            sample = df_tag.sample(number_to_sample)
            df = df.drop(list(sample.index))
            df_tag = df_tag.drop(list(sample.index))
            train = pd.concat([train, sample])
            
            number_to_sample = min(math.ceil(val_size*percentage), len(df_tag))
            sample = df_tag.sample(number_to_sample)
            df = df.drop(list(sample.index))
            val = pd.concat([val, sample])
            
    print(df.shape)
    if print_output:
        print_splits(train, val, test)

    return train, val, test


def stratified_split_by_page_id(df, splits, manual_test_tag=False, seed=0, print_output=False):
    if sum(splits) > 1:
        raise ValueError
        
    r = random.Random(seed)

    train = pd.DataFrame(columns=df.columns)
    val = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)
    
    page_ids = set(df['pid'])
    page_proportions_per_tag = df.groupby('tag') \
                                .agg(set)['pid'] \
                                .apply(lambda x: len(x)/len(page_ids)) \
                                .sort_values(ascending=False)
    
    test_size = int(splits[2]*len(page_ids))
    already_sampled_pids = set()
    for tag, percentage in page_proportions_per_tag.iteritems():  
        if manual_test_tag:
            df_tag = df.loc[~(df["auto_tag"]) & (df['tag'] == tag)]
        else:
            df_tag = df.loc[df['tag'] == tag]
        
        tag_pids = set(df_tag['pid']).difference(already_sampled_pids)
        missing = math.ceil(test_size*percentage) - len(set(test.loc[test['tag'] == tag, 'pid']))
        number_to_sample = min(max(0, missing), len(tag_pids))
        sampled_tag_pids = set(r.sample(list(tag_pids), number_to_sample))
        already_sampled_pids = already_sampled_pids.union(sampled_tag_pids)
        test = test.append(df[df['pid'].isin(sampled_tag_pids)])
        
    train_size = int(splits[0]*len(page_ids))
    val_size = len(page_ids) - train_size - test_size
    for tag, percentage in page_proportions_per_tag.iteritems():  
        df_tag = df.loc[df['tag'] == tag]
        tag_pids = set(df_tag['pid']).difference(already_sampled_pids)
        if len(tag_pids) > 0:
            number_to_sample = max(1, math.ceil(train_size*percentage) - len(set(train.loc[train['tag'] == tag, 'pid'])))
            sampled_tag_pids_train = set(r.sample(list(tag_pids), 
                                                 min(max(1, len(list(tag_pids)) - 1), number_to_sample)))
            already_sampled_pids = already_sampled_pids.union(sampled_tag_pids_train)
            train = train.append(df[df['pid'].isin(sampled_tag_pids_train)])

            tag_pids = set(df_tag['pid']).difference(already_sampled_pids)
            number_to_sample = max(1, math.ceil(val_size*percentage) - len(set(val.loc[val['tag'] == tag, 'pid'])))
            sampled_tag_pids_val = set(r.sample(list(tag_pids), 
                                                 min(len(list(tag_pids)), number_to_sample)))
            already_sampled_pids = already_sampled_pids.union(sampled_tag_pids_val)
            val = val.append(df[df['pid'].isin(sampled_tag_pids_val)])

    if print_output:
        print_splits(train, val, test)
            
    return train, val, test
    

def print_tag_distributions(df):
    tag_counts = df['tag'].value_counts()
    page_counts = df.groupby(['tag']).agg(list)['pid'].apply(lambda x: len(set(x)))
    issue_counts = df.groupby(['tag']).agg(list)['meta_issue_id'].apply(lambda x: len(set(x)))
    longest_tag = max(tag_counts.index.map(len))
    for tag, count in tag_counts.sort_index().iteritems():
        print("{:{longest_tag}} {:6} items ({:.1f}%) on {:4} pages on {:4} issues".format(tag, count, count/len(df)*100, page_counts[tag], issue_counts[tag], longest_tag=longest_tag))
    
    
def print_splits(train, val, test):
    
    df = pd.concat([train, val, test])
    
    print(f"Dataset: {df.shape[0]} items on {len(df['pid'].unique())} pages.")
    print(f"-- Train set: {train.shape[0]} items on {len(train['pid'].unique())} pages.")
    print(f"-- Validation set: {val.shape[0]} items on {len(val['pid'].unique())} pages.")
    print(f"-- Test set: {test.shape[0]} items on {len(test['pid'].unique())} pages.")
    