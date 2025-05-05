_base_ = ['/home/project/MMDet/mmdetection/configs/my_config/mask2former/mask2former_r50_8xb2-lsj-50e_coco-instance.py']


# experimental settings
load_from = '/home/project/host_workspace/mmdet/checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
batch_size = 8
num_workers = 1
work_dir = '/home/project/host_workspace/mmdet/mask2former_r50_finetune'

# mask2former settings
num_things_classes = 6
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (640, 640)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
model = dict(
    data_preprocessor=data_preprocessor,
    # panoptic_head=dict(
    #     num_things_classes=num_things_classes,
    #     num_stuff_classes=num_stuff_classes,
    #     loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    # panoptic_fusion_head=dict(
    #     num_things_classes=num_things_classes,
    #     num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'CocoDataset'
classes = ('Reflective', 'Verti-Edge', 'Corr-Shov-Disp', 'Rutt-Depress', 'Construction', 'Alligator')
# classes = ('반사균열', '세로방향균열, 단부균열', '코루게이션, 쇼빙, 밀림균열', '러팅, 함몰', '시공균열', '거북등')
data_root = '/home/project/MMDet/mmdetection/data/rddphcv/'

train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root, 
        ann_file='annotations/rddphcv_coco_fmt_instance_train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/rddphcv_coco_fmt_instance_test.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/rddphcv_coco_fmt_instance_test.json',
    classwise=True,
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = val_evaluator


