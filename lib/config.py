# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
import onnx
import numpy as np
import random
import os
from loguru import logger
from typing import List, Union, Dict


def fix_seed(identical_seed : int = 1024):
  logger.info('onnx version:{}'.format(onnx.__version__))
  logger.info('torch version:{}'.format(torch.__version__))
  logger.info('torch local:{}'.format(torch.__path__))

  torch.manual_seed(identical_seed)
  torch.cuda.manual_seed(identical_seed)
  torch.cuda.manual_seed_all(identical_seed)
  np.random.seed(identical_seed)
  random.seed(identical_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


class QConfig(object):
    def __init__(
        self, 
        model_name : str,
        input_dims : List[tuple],
        nn_module : torch.nn.Module, # define model and Load pretrained state_dcit
        has_residuals_module : bool,
        is_run_simpify : bool = False, 
        input_names : str = None,
        output_names : str = None,
        dynamic_axes : Dict[str, Dict[int, str]] = None,
        lonlp : List[str] = [],
        ignore_policy : Union[str, List[str]] = None,
        onnx_out_dir : str = None, 
        calib_out_dir : str = None, 
        calib_batch_size : int = 4,
        calib_num_workers : int = 12,
        calibrator : str = 'histogram',  # or 'max'
        hist_percentile : List[np.float32] = [99.9, 99.99, 99.999, 99.9999]):

        self.model_name = model_name   # 'resnet50'
        self.input_dims = input_dims   # [(1, 3, 512, 512),]
        self.nn_module = nn_module
        self.has_residuals_module = has_residuals_module  
        self.is_run_simpify = is_run_simpify 

        self.input_names = input_names   # ['input',]
        self.output_names = output_names # ['output',]
        self.dynamic_axes = dynamic_axes # {"input": {0: "batch"}, "output": {0: "batch"}}

        # Layers of non learning parameters, but can quantize, like avgpool layer. 
        # User must set the layer name according to the forward data flow !!!
        self.lonlp = lonlp      
        self.ignore_policy = ignore_policy  
        self.onnx_out_dir = onnx_out_dir
        self.calib_out_dir = calib_out_dir

        self.calib_batch_size = calib_batch_size
        self.calib_num_workers = calib_num_workers
        self.calibrator = calibrator
        self.hist_percentile = hist_percentile

        self.finetune_epochs = 10         

        ########################################
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_module = self.nn_module.to(self.device)
    
        if self.onnx_out_dir is None:
            self.onnx_out_dir = 'unit_test/' + model_name 

        if self.calib_out_dir is None:
            self.calib_out_dir  = self.onnx_out_dir 

        if self.onnx_out_dir[-1] != '/':
            self.onnx_out_dir = self.onnx_out_dir + "/"

        if self.calib_out_dir[-1] != '/':
            self.calib_out_dir = self.calib_out_dir + "/"

        if not os.path.exists(self.onnx_out_dir):
            os.makedirs(self.onnx_out_dir) 

        if not os.path.exists(self.calib_out_dir):
            os.makedirs(self.calib_out_dir)    

        self.nquant_onnx = self.onnx_out_dir + self.model_name + "_nquant.onnx"
        self.quant_onnx_before_calib = self.onnx_out_dir + self.model_name + "_quant_before_calib.onnx"
        self.quant_onnx_after_calib = self.onnx_out_dir + self.model_name + "_quant_after_calib.onnx"
    
        ########################################


    def summary(self):
        logger.info("model name:{}".format(self.model_name))
        logger.info("model input_dims:{}".format(self.input_dims))
        logger.info("model input_names:{}".format(self.input_names))
        logger.info("model output_names:{}".format(self.output_names))
        logger.info("model dynamic_axes:{}".format(self.dynamic_axes))
        logger.info("model lonlp:{}".format(self.lonlp))
        logger.info("model ignore_policy:{}".format(self.ignore_policy))
        logger.info("model has_residuals_module:{}".format(self.has_residuals_module))
        logger.info("model onnx_out_dir:{}".format(self.onnx_out_dir))
        logger.info("model calib_batch_size:{}".format(self.calib_batch_size))
        logger.info("model calib_num_workers:{}".format(self.calib_num_workers))
        logger.info("model calib_out_dir:{}".format(self.calib_out_dir))
        logger.info("model calibrator:{}".format(self.calibrator))
        logger.info("model hist_percentile:{}".format(self.hist_percentile))
        logger.info("model finetune_epochs:{}".format(self.finetune_epochs))

    def check(self):
        def check_str(it) -> bool:
            if it is None or it == '':
                logger.error("check {} error".format(it))
                return False
            return True    

        def check_list(it) -> bool:
            if it is None or len(it) == 0:
                logger.error("check {} error".format(it))
                return False
            return True 

        def check_uint(it) -> bool:
            if it is None or it <= 0:
                logger.error("check {} error".format(it))
                return False
            return True 

        ret = check_str(self.model_name) 
        ret = ret and check_str(self.onnx_out_dir)         
        ret = ret and check_list(self.input_dims) 
        ret = ret and check_uint(self.calib_batch_size) 
        ret = ret and check_uint(self.calib_num_workers) 
        ret = ret and check_uint(self.finetune_epochs) 
        if not ret:
            exit(0)


#
# class OutConv(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# model = OutConv(3, 6)

# t = QConfig("resnet", [(1,3,244,244)], model, False)
# t.summary()
# t.check()
#
