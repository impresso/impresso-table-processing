_base_ = ['../../../../../../../home/amvernet/mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py',
          '../../../../../../../home/amvernet/mmdetection/configs/_base_/default_runtime.py']

batch_size = 2
classes = ('table',)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=1e-6)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='exp', gamma=0.85, by_epoch=True)
evaluation=dict(classwise=True, metric=['bbox', 'segm'], save_best="segm_mAP")
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(interval=1,
                         max_keep_ckpts=4,
                         by_epoch=True,
                         out_dir=run_path)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800)],
        keep_ratio=False,
        ratio_range=(0.9, 1.1)),
    dict(type='Rotate', level=10, max_rotate_angle=1, prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
img_prefix = "/scratch/students/amvernet/dataset/images"
data = {
    'samples_per_gpu': batch_size,
    'workers_per_gpu': 4,
    'train': {
        'type': dataset_type,
        'classes': classes,
        'ann_file': f'{experiment_path}/coco_annotations_{dataset_name}_train.json',
        'img_prefix': img_prefix,
        'pipeline': train_pipeline},
    'val': {
        'type': dataset_type,
        'classes': classes,
        'ann_file': f'{experiment_path}/coco_annotations_{dataset_name}_val.json',
        'img_prefix': img_prefix,
        'pipeline': test_pipeline},
    'test': {
        'type': dataset_type,
        'classes': classes,
        'ann_file': f'{experiment_path}/coco_annotations_{dataset_name}_test.json',
        'img_prefix': img_prefix,
        'pipeline': test_pipeline}}

model = {
    'roi_head': {
        'bbox_head': {
            'num_classes': len(classes)},
        'mask_head': {
            'num_classes': len(classes)}}}

load_from = '/scratch/students/amvernet/data/checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(
             project=experiment_name + "_mmDetection",
             config=None,
             name=run_name,
             dir=run_path))])
