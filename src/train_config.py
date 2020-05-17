"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CITYSCAPES_DIR=os.environ.get('CITYSCAPES_DIR')

args = dict(

    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='./exp_fine',
    resume_path='./exp/checkpont.pth', 

    train_dataset = {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': "/data/gtFine_trainvaltest",
            'type': 'train',
            'size': 3000,
            'class_id': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                       'keys': ('image', 'instance','label'),
                       'size': (1024,1024),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 8,
        'workers': 8
    }, 

    val_dataset = {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': "/data/gtFine_trainvaltest",
            'type': 'val',
            'class_id': None,
            'size': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 8,
        'workers': 8
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [4,8]
        }
    }, 

    lr=5e-5,
    n_epochs=50,

    # loss options
    loss_opts={
        'to_center': False,
        'n_sigma': 2,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 0,
        'w_seed': 0,
    },
)


def get_args():
    return copy.deepcopy(args)
