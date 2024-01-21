# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
from onnxsim import simplify
import onnx, os, sys  
from loguru import logger

sys.path.insert(0, "../pytorch-quantization_v2.1.2") #v2.1.0

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as __quant_nn

quant_nn = __quant_nn
quant_desc_input = QuantDescriptor(calib_method="histogram")
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantConv2d_WeightOnly.set_default_quant_desc_input(quant_desc_input)

def export_onnx_quant(model, inputs, onnx_name):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, inputs, onnx_name, opset_version=13)

    model = onnx.load(onnx_name)
    os.remove(onnx_name)

    onnx.checker.check_model(model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, onnx_name)
    logger.info("simplify onnx done !")

    quant_nn.TensorQuantizer.use_fb_fake_quant = False


def export_onnx(model, inputs, onnx_name):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, inputs, onnx_name, opset_version=13)

    model = onnx.load(onnx_name)
    os.remove(onnx_name)

    onnx.checker.check_model(model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, onnx_name)
    logger.info("simplify onnx done !")


def summary(model, input_dims, depth=3):
    from torchsummary import summary
    summary(model, input_data=input_dims, depth=depth) 
