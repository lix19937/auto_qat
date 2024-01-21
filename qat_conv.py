# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from __models.conv import Conv
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(
      os.path.basename(__file__).split('.')[0], 
      [(4, 6, 224, 224)], 
       Conv(in_channels=6, out_channels=10), 
      False) 
 
    pipeline(cfg)
