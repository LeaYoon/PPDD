
import argparse
import os
import os.path as osp  #ADD

import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


### start of import for vis-mmdet
import cv2
import pickle

import numpy as  np

import mmcv
from mmseg.apis import init_model, inference_model
from mmengine.structures import BaseDataElement, InstanceData
from mmengine.structures import PixelData #ADD

from mmengine.visualization import Visualizer
from mmseg.visualization import SegLocalVisualizer #ADD
from mmseg.structures import SegDataSample #ADD

# from mmdet.registry import VISUALIZERS
from mmseg.registry import VISUALIZERS
def main(
        config,
        checkpoint,
        data_path,
        project,
        dir_name,
        device='cpu'
):
    # set device
    device = 'cuda:0' if device == 'gpu' else 'cpu'

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)
    
    ### vis-mmdet setting
    # visualizer_cfg_path = r"/home/project/MMSeg/mmsegmentation/visualization_mmdet/vis_cfg.pkl"
    dataset_meta_path = r"/home/project/MMSeg/mmsegmentation/visualization_mmdet/dataset_meta.pkl"
    
    # with open(visualizer_cfg_path, 'rb') as f:
    #     visualizer_cfg = pickle.load(f)

    with open(dataset_meta_path, 'rb') as f:
        dataset_meta = pickle.load(f)
    # dataset_meta['classes'] = ('Reflection', 'Longi-Edge', 'Corr-Shov-Slip', 'Rutt-Depress', 'Pothole-Ravel-Strip', 'Construction', 'Alligator', 'Non-crack')
    dataset_meta['classes'] = ('RC', 'LEC', 'CSSC', 'RDC', 'Pothole-Ravel-Strip', 'CJC', 'AC', 'Non-crack')
    dataset_meta['palette'] = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 11, 123), (106, 0, 228), (0, 60, 100), (0, 0, 0)]

    output_dir = os.path.join(project, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # visualizer.dataset_meta = dataset_meta
    # visualizer.vis_backends = [dict(type='LocalVisBackend')]
    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=output_dir
    ) 
    seg_local_visualizer.dataset_meta = dataset_meta
    visualizer = seg_local_visualizer


    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = mmcv.imread(img_path, channel_order='rgb')

        output = inference_model(model, img)
        visualizer.add_datasample(
            name='new_result',
            image=img,
            data_sample=output,
            draw_gt=False,
            draw_pred=True,
            show=False
        )

        save_path = os.path.join(output_dir, img_name)
        drawn_img = visualizer.get_image()
        cv2.imwrite(save_path, cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR))
    return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--data_path', help='the directory for model to predict')
    parser.add_argument('--project', help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--dir_name', help='directory name where painted images will be saved')
    parser.add_argument('--device', default='cpu', help='device used for inference')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    print("visualize mmdet model")
    opt = parse_opt()
    main(**vars(opt))