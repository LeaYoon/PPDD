# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pickle

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo
    print("MMSEG DATASET META: ", dataset.metainfo)
    print()
    dataset_meta_path = r"/home/project/MMSeg/mmsegmentation/visualization_mmdet/dataset_meta.pkl"
    with open(dataset_meta_path, 'rb') as f:
        dataset_meta = pickle.load(f)
    
    # dataset_meta['classes'] = ('Reflective', 'Verti-Edge', 'Corr-Shov-Disp', 'Rutt-Depress', 'Pothole-Ravel-Strip', 'Construction', 'Alligator', 'Non-crack')
    dataset_meta['classes'] = ('RC', 'LEC', 'CSSC', 'RDC', 'Pothole-Ravel-Strip', 'CJC', 'AC', 'Non-crack')
    dataset_meta['palette'] = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (0, 11, 123), (106, 0, 228), (0, 60, 100), (0, 0, 0)]
    visualizer.dataset_meta = dataset_meta
    print("MMDet DATASET META: ", dataset_meta)
    print()

    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        img = img[..., [2, 1, 0]]  # bgr to rgb
        data_sample = item['data_samples'].numpy()
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None
        img = img.copy() # cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
        visualizer.add_datasample(
            name=osp.basename(img_path),
            image=img,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=args.show_interval,
            out_file=out_file,
            show=not args.not_show)
        progress_bar.update()

if __name__ == '__main__':
    main()
