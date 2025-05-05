# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pickle

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--only_mask', default=False, action='store_true', help='output only mask')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo
    dataset_meta_path = r"/home/project/MMSeg/mmsegmentation/visualization_mmdet/dataset_meta.pkl"
    
    # with open(visualizer_cfg_path, 'rb') as f:
    #     visualizer_cfg = pickle.load(f)

    with open(dataset_meta_path, 'rb') as f:
        dataset_meta = pickle.load(f)

    # dataset_meta['classes'] = ('RC', 'LEC', 'CSSC', 'RDC', 'Pothole-Ravel-Strip', 'CJC', 'AC', 'Non-crack')
    # dataset_meta['palette'] = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 11, 123), (106, 0, 228), (0, 60, 100), (0, 0, 0)]
    dataset_meta['classes'] = ('RC', 'LEC', 'CSSC', 'RDC', 'CJC', 'AC')
    dataset_meta['palette'] = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)]
    visualizer.dataset_meta = dataset_meta
    print("MMDet DATASET META: ", dataset_meta)
    print()


    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb
        gt_bboxes = gt_instances.get('bboxes', None)
        if args.only_mask:
            num_instances = gt_instances.bboxes.shape[0]
            if gt_bboxes is not None:
                dtype = gt_bboxes.tensor.dtype
                device = gt_bboxes.tensor.device
            else:
                dtype = torch.float32
                device = torch.device('cpu')
            gt_instances.bboxes = torch.zeros((num_instances, 4), dtype=dtype, device=device)
        elif gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()
