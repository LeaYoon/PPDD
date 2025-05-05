from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class RDDPHCVDataset(BaseSegDataset):
    METAINFO = dict(
        classes = ('Reflective', 'Verti-Edge', 'Corr-Shov-Disp', 'Rutt-Depress', 
                'Pothole-Ravel-Strip', 'Construction', 'Alligator', 'Non-crack'), 
        palette = [(128, 128, 128), (129, 127, 38), (120, 69, 125), (53, 125, 34), 
          (0, 11, 123), (118, 20, 12), (122, 81, 25), (0, 0, 0)])
    def __init__(self, 
                img_suffix='.PNG',
                seg_map_suffix='.PNG',
                **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)