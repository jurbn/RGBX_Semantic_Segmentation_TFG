import os
import os.path as osp
import sys
import time
import numpy as np
import seaborn as sns
from easydict import EasyDict as edict
import argparse
from pycocotools.coco import COCO

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'lindenthal-camera-traps'
C.dataset_path = osp.join(C.root_dir, 'data', C.dataset_name)
C.random_scramble = True

# COCO JSON files for each split
C.train_json = osp.join(C.dataset_path, 'lindenthal_coco', 'train.json')
C.val_json = osp.join(C.dataset_path, 'lindenthal_coco', 'val.json') # annotations_lindenthal

# RGB and Additional Modality (e.g., Depth or Thermal) Folder Settings
C.rgb_root_folder = osp.join(C.dataset_path, 'lindenthal_coco', 'images')
C.rgb_format = '.jpg'
C.x_root_folder = osp.join(C.dataset_path, 'lindenthal_coco', 'images')  # Adjust to the actual modality folder name
C.x_format = '.png'#'.exr'  # Change format to your additional modality format
C.x_is_single_channel = True  # Set True if using a single-channel additional modality

# Dynamically read class names and count from COCO JSON
train_coco = COCO(C.train_json)
C.class_names = ['background'] + [cat['name'] for cat in train_coco.loadCats(train_coco.getCatIds())]
C.num_classes = len(C.class_names)
C.class_colors = [
    [0, 0, 0],  # Background
]  + sns.color_palette(None, len(C.class_names))
C.class_colors = [[int(255 * x) for x in rgb] for rgb in C.class_colors]


"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0., 0., 0.], dtype=np.float32)
C.norm_std = np.array([1., 1., 1.],  dtype=np.float32)

"""Settings for network"""
C.backbone = 'mit_b0'
C.pretrained_model = osp.join(C.root_dir, 'pretrained', 'segformers', 'mit_b0.pth')
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 1e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.dropout_rate = 0.5
C.modality_dropout = True
C.batch_size = 16
C.num_train_imgs = len(train_coco.getImgIds())
C.num_eval_imgs = len(COCO(C.val_json).getImgIds())  # Dynamic count based on COCO data
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 12
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [480, 640]

"""Store Config"""
C.checkpoint_start_epoch = 0
C.checkpoint_step = 10

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('out/log_' + C.dataset_name + '_' + C.backbone + '_' + 'pretrained_jet_dropout')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = osp.join(C.log_dir, f'log_{exp_time}.log')
C.link_log_file = osp.join(C.log_dir, 'log_last.log')
C.val_log_file = osp.join(C.log_dir, f'val_{exp_time}.log')
C.link_val_log_file = osp.join(C.log_dir, 'val_last.log')

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument('-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
