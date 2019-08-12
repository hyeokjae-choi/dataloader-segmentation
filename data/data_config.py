import os.path as osp
from easydict import EasyDict as edict

from config import DATASET_DIR


__C = edict()

__C.SUALAB = edict()

__C.SUALAB.Crethi = edict(
    ds_dir=osp.join(DATASET_DIR, osp.join('055_crethi', 'bearing', 'original')),
    anno_fname='single_2class.json',
    imgset_name=osp.join('single_2class', 'ratio', '100%'),
    imgset_number=1,
    wrong_annotation_list=['NG_023.png',
                           'NG_086.png',
                           'NG_090.png',
                           'NG_091.png',
                           'NG_093.png',
                           'NG_097.png'],
)
__C.SUALAB.Leather = edict(
    ds_dir=osp.join(DATASET_DIR, osp.join('999_project', '004_edge_detection', 'leather', 'original')),
    anno_fname='single_2class.json',
    imgset_name='single_2class',
    imgset_number=1,
)

DATASET_CONFIG = __C