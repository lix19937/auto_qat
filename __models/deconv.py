
import torch
import torch.nn as nn
import numpy as np

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

quant_modules.initialize()

quant_desc_input = QuantDescriptor(calib_method='max')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(192, 100, 4, stride=2, padding=1, bias=False, output_padding=0)
        construct_w = np.ones((200, 100, 4, 4), dtype=np.float32)
        construct_w[:, -1, -1, -1] = 127
        self.deconv.weight = nn.Parameter(torch.tensor(construct_w))

    def forward(self, x):
        out = self.deconv(x)
        return out


def collect_stats(model, data, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    model(data)

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    # model.cuda()


if __name__ == "__main__":
    model = Net()
    model.eval()

    dummy_input = torch.tensor(np.tile(np.ones(120).reshape(1, 1, 1, 120), (200, 68, 1)).astype(np.float32))
    dummy_input[-1, -1, -1, -1] = 127

    data = dummy_input
    with torch.no_grad():
        collect_stats(model, data, num_batches=1)
        compute_amax(model, method="percentile", percentile=100)

    out = model(dummy_input)
    pytorch_out = out[:]

    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    torch.onnx.export(
        model, dummy_input, "test.onnx", verbose=True, opset_version=13, enable_onnx_checker=False)


    # import sys
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
    # from lib.fx import replace_residual_add_to_module   
    # model = replace_residual_add_to_module(model)

    # from lib.quantize import replace_to_quantization_module 
    # replace_to_quantization_module(model) 

    # onnx_name = os.path.basename(__file__).split('.')[0] + "_v2.onnx"  
    # export_onnx_quant(model, inputs, onnx_name)