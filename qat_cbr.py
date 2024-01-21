# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from __models.conv_bn_relu import CBR
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(
      os.path.basename(__file__).split('.')[0], 
      [(1, 3, 224, 224)], 
      CBR(n_channels=3, n_classes=2, bilinear=True), 
      False,
     # ignore_policy = ['outc.conv']
      ) 
 
    pipeline(cfg)