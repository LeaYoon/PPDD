_base_ = [
    '/home/project/MMSeg/mmsegmentation/configs/my_config/datasets/rddphcv.py',
    '/home/project/MMSeg/mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '/home/project/MMSeg/mmsegmentation/configs/_base_/schedules/schedule_80k.py'
    ]

# model settings by deeplabv3plus_r50-d8_4xb2-40k_cityscapes-769x769.py + ...80k....py
crop_size = (640, 640) # (769, 769)

### custum settings
num_classes = 8

# fine-tuning
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_cityscapes_20201226_082915-1b9a2f6f.pth'  # noqa
checkpoint = '/home/project/host_workspace/mmseg/checkpoints/deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth'

# ## resume: schedule 32000 -> 80000
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=48000, val_interval=8000)
# load_from = '/home/project/host_workspace/mmseg/deeplabv3plus/iter_32000.pth'

## model
# model = dict(
#     # pretrained='torchvision://resnet50',
#     # backbone=dict(type='ResNet'),
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
#     ),
#     data_preprocessor=dict(size=crop_size),
#     decode_head=dict(align_corners=True, num_classes = num_classes),
#     auxiliary_head=dict(align_corners=True, num_classes = num_classes),
#     test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

# r101
# model = dict(
#     # pretrained='open-mmlab://resnet101_v1c', 
#     backbone=dict(
#         init_cfg=dict(checkpoint=checkpoint)),
#     data_preprocessor=dict(size=crop_size),
#     decode_head=dict(align_corners=True, num_classes = num_classes),
#     auxiliary_head=dict(align_corners=True, num_classes = num_classes),
#     test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
model = dict(
    pretrained=checkpoint,
    # backbone=dict(
        # init_cfg=dict(checkpoint=checkpoint)),
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(align_corners=True, num_classes = num_classes),
    auxiliary_head=dict(align_corners=True, num_classes = num_classes),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(513, 513)))

dataset_type = 'RDDPHCVDataset'
data_root = '/home/project/host_workspace/dataset/MM_RDDPHCV_CAR'
batch_size = 8
num_workers = 1
work_dir = '/home/project/host_workspace/mmseg/deeplabv3plus_3_finetune'

# same with rddphcv.py
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers, # 2
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1, # 4
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator