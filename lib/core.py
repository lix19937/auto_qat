# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

from quantize import replace_to_quantization_module, apply_custom_rules_to_quantizer, calibrate_model, export_onnx_nquant, export_onnx_quant 
from fx import replace_residual_add_to_module   
from dataloader import get_dataloader, DataBase   

from loguru import logger

def pipeline(cfg) -> None:
    model_name = cfg.model_name
    model = cfg.nn_module
    dims = cfg.input_dims
    device = cfg.device

    cfg.summary()
    cfg.check()

    # Just as QAT benchmark
    export_onnx_nquant(model, file=cfg.nquant_onnx, size=dims,
      input_names=cfg.input_names, output_names=cfg.output_names, dynamic_axes=cfg.dynamic_axes)
    logger.info("#1 export_onnx_nquant done")

    # For residuals network
    if cfg.has_residuals_module:
        model = replace_residual_add_to_module(model)

    # Auto QDQ node insert, based on torch.nn.Module level
    replace_to_quantization_module(model, cfg.ignore_policy)
    logger.info("#2 replace_to_quantization_module done")

    # Custom rules for binding scale, based on onnx level, here we donot use dynamic shape
    apply_custom_rules_to_quantizer(model, export_onnx_quant, dims, lonlp=cfg.lonlp)
    logger.info("#3 apply_custom_rules_to_quantizer done")

    # logger.info(model.state_dict().keys())
    
    # QAT onnx for debug 
    export_onnx_quant(model, file=cfg.quant_onnx_before_calib, size=dims, run_simpify=cfg.is_run_simpify)
    logger.info("#4 export_onnx_quant before calib done")
    # exit(0)

    # DataSet
    dataset = DataBase(dims)
    logger.info("#5 dataset done")

    data_loader, num_calib_batch = get_dataloader(dataset, batch_size=cfg.calib_batch_size, num_workers=cfg.calib_num_workers)
    logger.info("#6 get_dataloader done")

    calibrate_model(model, model_name, data_loader, device, num_calib_batch, 
      calibrator=cfg.calibrator, hist_percentile=cfg.hist_percentile, out_dir=cfg.calib_out_dir)
    logger.info("#7 get_dataloader done")   

    # Check onnx  like:/usr/src/tensorrt/bin/trtexec --onnx=qat_resnet50_quant_after_calib.onnx --best --verbose
    export_onnx_quant(model, file=cfg.quant_onnx_after_calib, size=dims, run_simpify=cfg.is_run_simpify)
    logger.info("#8 export_onnx_quant after calib done")
    logger.info(model.state_dict().keys())
    for k,v in model.state_dict().items():
      if "_input_quantizer" in k:
        print(k, "|",  v )
    
    print(k, "|",  model )

    # Finetune
    logger.info("#9 finetune done")
