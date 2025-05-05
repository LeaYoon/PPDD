# mask2former_swin_s_250224_nojitter
python ./tools/visualize.py \
    --config /home/project/MMDet/mmdetection/configs/my_config/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_rddphcv.py \
    --checkpoint /home/project/host_workspace/mmdet/mask2former_swin_s_finetune2/best_coco_segm_mAP_iter_368750.pth \
    --data_path /home/project/host_workspace/data/test/car/polygon/images \
    --project /home/project/host_workspace/mmdet/outputs \
    --dir_name mask2former_swin_s_250502_nojitter \
    --device cuda:0 \
    --only_mask

# mask2former_
# python ./tools/visualize.py \
#     --config /home/project/host_workspace/mmdet/mask2former_r101_finetune3/mask2former_r101_8xb2-lsj-50e_rddphcv.py \
#     --checkpoint /home/project/host_workspace/mmdet/mask2former_r101_finetune3/best_coco_segm_mAP_iter_323750.pth \
#     --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images \
#     --project /home/project/host_workspace/mmdet/outputs \
#     --dir_name mask2former_r101_250325_nojitter \
#     --device cuda:0 \
#     --only_mask