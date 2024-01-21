# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import sys

sys.path.append('./__models/yolov7')

from __models.yolov7.yolov7_warpper import load_model
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(os.path.basename(__file__).split('.')[0], [(1, 3, 640, 640)], load_model(weight="__models/yolov7/yolov7.pt", device="cuda:0"), False) 
 
    pipeline(cfg)
