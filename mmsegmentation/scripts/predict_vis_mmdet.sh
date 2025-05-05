# swinl
CUDA_VISIBLE_DEVICES=0,1 python ./tools/predict_vis_mmdet.py --config /home/project/host_workspace/mmseg/mask2former_finetune/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_rddphcv.py --checkpoint /home/project/host_workspace/mmseg/mask2former_finetune/best_mIoU_iter_90000.pth --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images --project /home/project/host_workspace/mmseg/outputs --dir_name mask2former_swinl_250325_test_sub1 --device cuda:0

# # r101
CUDA_VISIBLE_DEVICES=0,1 python ./tools/predict_vis_mmdet.py --config /home/project/host_workspace/mmseg/mask2former_r101_finetune/mask2former_r101_8xb2-90k_rddphcv.py --checkpoint /home/project/host_workspace/mmseg/mask2former_r101_finetune/best_mIoU_iter_85000.pth --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images --project /home/project/host_workspace/mmseg/outputs --dir_name mask2former_r101_250325_test_sub1 --device cuda:0

# segformer
CUDA_VISIBLE_DEVICES=0,1 python ./tools/predict_vis_mmdet.py --config /home/project/host_workspace/mmseg/segformer_2/segformer_mit-b5_8xb2-160k_rddphcv.py --checkpoint /home/project/host_workspace/mmseg/segformer_2/iter_144000.pth --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images --project /home/project/host_workspace/mmseg/outputs --dir_name segformer_250325_test_sub1 --device cuda:0

# deeplabv3+
CUDA_VISIBLE_DEVICES=0,1 python ./tools/predict_vis_mmdet.py --config /home/project/host_workspace/mmseg/deeplabv3plus_3_finetune/deeplabv3plus_r50b-d8_4xb2-80k_rddphcv.py --checkpoint /home/project/host_workspace/mmseg/deeplabv3plus_3_finetune/iter_80000.pth --data_path /home/project/host_workspace/data/test_sub1/car/polygon/images --project /home/project/host_workspace/mmseg/outputs --dir_name deeplabv3p_250325_test_sub1 --device cuda:0
