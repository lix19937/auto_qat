
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model 
from datasets.dataset_factory import get_dataset


def main(opt):
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  print("aarch:", opt.arch)
  print("heads:", opt.heads)
  print("hconv:", opt.head_conv)

  model = create_model(opt.arch, opt.heads, opt.head_conv)
  return model


def load_model(arch = "hourglass", heads = {'hm': 80, 'wh': 2, 'reg': 2}, head_conv = 64):
  model = create_model(arch, heads, head_conv)
  return model

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
  
  load_model()

# python3 centernet_warpper.py ctdet --arch hourglass  --dataset  coco
# python3 centernet_warpper.py ctdet --arch hourglass  --dataset  kitti
# python3 centernet_warpper.py ctdet --arch hourglass  --dataset coco_hp 
# python3 centernet_warpper.py ctdet --arch hourglass  --dataset pascal
