# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from __models.large_hourglass import HourglassNet2d
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(
      os.path.basename(__file__).split('.')[0], 
      [(1, 256, 256, 128)], 
      HourglassNet2d(num_stacks=1), 
      True) 
 
    pipeline(cfg)
