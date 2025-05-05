_base_ = ['/home/project/MMDet/mmdetection/configs/my_config/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_rddphcv.py']

### custom
# load_from = '/home/project/host_workspace/mmdet/checkpoints/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'  # noqa
load_from = '/home/project/host_workspace/mmdet/mask2former_swin_s_finetune/iter_10000.pth'
batch_size = 2 # 64 on 2 GPU # 8 on 1 GPU
num_workers = 1
work_dir = '/home/project/host_workspace/mmdet/mask2former_swin_s_finetune2'

ataset_type = 'CocoDataset'
classes = ('Reflective', 'Verti-Edge', 'Corr-Shov-Disp', 'Rutt-Depress', 'Construction', 'Alligator')
data_root = '/home/project/MMDet/mmdetection/data/rddphcv/'

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='coco/segm_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

### original 
depths = [2, 2, 18, 2]
# model = dict(
#     backbone=dict(
#         depths=depths, 
#         init_cfg=dict(type='Pretrained', checkpoint=load_from)))
model = dict(
    backbone=dict(
        depths=depths))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
