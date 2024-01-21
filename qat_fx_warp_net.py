# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from __models.fx_warp_net import SimpWarp_M
from lib.core import pipeline
from lib.config import QConfig, fix_seed
import os

if __name__ == "__main__":
    fix_seed()

    cfg = QConfig(os.path.basename(__file__).split('.')[0], [(1, 4, 244, 244)], SimpWarp_M(4, 4), True) 
 
    pipeline(cfg)

    