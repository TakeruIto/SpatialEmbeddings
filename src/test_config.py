"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CITYSCAPES_DIR="/gs/hs0/tga-shinoda/16B01730/data/gtFine_trainvaltest"

args = dict(

    cuda=True,
    display=False,

    save=True,
    save_dir='./masks',
    checkpoint_path='./exp_all5e-4_cropobj/best_iou_model.pth',

    dataset= { 
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'val',
            'class_id': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [4, 8],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
