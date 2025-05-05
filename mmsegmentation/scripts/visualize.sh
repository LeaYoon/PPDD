# mask2former_swin_s_250224_nojitter
# python ./tools/visualize.py \
#     --config /home/project/MMDet/mmdetection/configs/my_config/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_rddphcv_testsub.py \
#     --checkpoint /home/project/host_workspace/mmdet/mask2former_swin_s_finetune2/best_coco_segm_mAP_iter_368750.pth \
#     --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images \
#     --project /home/project/host_workspace/mmdet/outputs \
#     --dir_name mask2former_swin_s_250224_nojitter \
#     --device cuda:0 \
#     --only_mask

# mask2former_swin-l
python /home/project/MMDet/mmdetection/tools/visualize.py \
    --config /home/project/host_workspace/mmseg/mask2former_finetune/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_rddphcv.py \
    --checkpoint /home/project/host_workspace/mmseg/mask2former_finetune/best_mIoU_iter_90000.pth \
    --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images \
    --project /home/project/host_workspace/mmdet/outputs/sem_outputs \
    --dir_name mask2former_swinl_250313 \
    --device cuda:0