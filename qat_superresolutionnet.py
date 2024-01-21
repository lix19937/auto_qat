# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from __models.superresolutionnet import SuperResolutionNet
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(os.path.basename(__file__).split('.')[0], [(1, 1, 244, 244)], SuperResolutionNet(upscale_factor=3), False) 
 
    pipeline(cfg)
