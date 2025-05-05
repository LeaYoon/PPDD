_base_ = ['/home/project/MMDet/mmdetection/configs/my_config/mask2former/mask2former_r50_8xb2-lsj-50e_rddphcv.py']

# experimental settings
# load_from = '/home/project/host_workspace/mmdet/checkpoints/mask2former_r101_8xb2-lsj-50e_coco_20220426_100250-ecf181e2.pth'
load_from = '/home/project/host_workspace/mmdet/mask2former_r101_finetune2/iter_5000.pth'
batch_size = 16 # 64 on 2 GPU # 8 on 1 GPU
num_workers = 1
work_dir = '/home/project/host_workspace/mmdet/mask2former_r101_finetune3'

num_things_classes = 6
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (640, 640)

dataset_type = 'CocoDataset'
classes = ('Reflective', 'Verti-Edge', 'Corr-Shov-Disp', 'Rutt-Depress', 'Construction', 'Alligator')
data_root = '/home/project/MMDet/mmdetection/data/rddphcv/'

# resume
train_cfg = dict(type='IterBasedTrainLoop', max_iters=323750, val_interval=5000)
# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=num_workers)

# learning policy
max_iters = 323750
param_scheduler = dict(
    type='MultiStepLR',
    begin=45000,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)
auto_scale_lr = dict(enable=False, base_batch_size=64) # 16

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint=load_from)))


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='coco/segm_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
