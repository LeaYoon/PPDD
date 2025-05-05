import os
import argparse
import pickle

import cv2
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import BaseDataElement, InstanceData

from mmdet.registry import VISUALIZERS

def convert_to_BaseDataElement(pred_bboxes, pred_labels, pred_scores, pred_masks, only_masks=False, only_bboxes=False):
    if only_masks and only_bboxes :
        raise ValueError("Either only_masks or only_bboxes should be True")
    data_sample = BaseDataElement()
    data_sample.gt_instances = InstanceData()
    data_sample.pred_instances = InstanceData()    

    if not isinstance(pred_bboxes, torch.Tensor):
        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
        pred_labels = torch.tensor(pred_labels, dtype=torch.int64)
        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
        pred_masks = torch.tensor(pred_masks, dtype=torch.float32)
    
    data_sample.pred_instances.labels = torch.tensor(pred_labels, dtype=torch.int64)
    data_sample.pred_instances.scores = torch.tensor(pred_scores, dtype=torch.float32)
    if not only_bboxes and not only_masks:
        data_sample.pred_instances.bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
        data_sample.pred_instances.masks = torch.tensor(pred_masks, dtype=torch.float32)
        return data_sample
    elif only_bboxes:
        data_sample.pred_instances.bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
        return data_sample
    elif only_masks:
        data_sample.pred_instances.masks = torch.tensor(pred_masks, dtype=torch.float32)
        return data_sample
    
def main(
        config,
        checkpoint,
        data_path,
        project,
        dir_name,
        device='cpu',
        only_mask=False,
        only_bbox=False
):
    # set device
    device = 'cuda:0' if device == 'gpu' else 'cpu'

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)

    # visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.vis_backends = [dict(type='LocalVisBackend')]

    # visualizer.dataset_meta = dataset.metainfo
    dataset_meta_path = r"/home/project/MMSeg/mmsegmentation/visualization_mmdet/dataset_meta.pkl"
    
    # with open(visualizer_cfg_path, 'rb') as f:
    #     visualizer_cfg = pickle.load(f)

    with open(dataset_meta_path, 'rb') as f:
        dataset_meta = pickle.load(f)
    # class names and colors
    dataset_meta['classes'] = ('RC', 'LEC', 'CSSC', 'RDC', 'CJC', 'AC')
    dataset_meta['palette'] = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)]
    visualizer.dataset_meta = dataset_meta
    print("MMDet DATASET META: ", dataset_meta)
    print()

    output_dir = os.path.join(project, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = mmcv.imread(img_path, channel_order='rgb')

        output = inference_detector(model, img)    

        # remove bbox or mask fields
        if "bboxes" in output.pred_instances.keys():
            pred_bboxes = output.pred_instances.bboxes
        else:
            pred_bboxes = None
        if "masks" in output.pred_instances.keys():
            pred_masks = output.pred_instances.masks
        else:
            pred_masks = None
        pred_labels = output.pred_instances.labels
        pred_scores = output.pred_instances.scores


        output = convert_to_BaseDataElement(pred_bboxes, pred_labels, pred_scores, pred_masks, only_masks=only_mask, only_bboxes=only_bbox)

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
    parser.add_argument('--only_mask', action='store_true', help='visualize only masks')
    parser.add_argument('--only_bbox', action='store_true', help='visualize only bounding boxes')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    print("visualize mmdet model")
    opt = parse_opt()
    main(**vars(opt))
