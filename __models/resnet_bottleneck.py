# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
import torch.nn.functional as F

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True, quantize: bool = False):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        if quantize:
            self.conv = quant_nn.QuantConv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                padding, groups=groups, bias=False)

        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU(inplace=True) if activation else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, down_sample: bool=False, groups=1, quantize: bool = False):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False, quantize=quantize) \
            if in_channels != out_channels else torch.nn.Identity()

        self.conv = torch.nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1, quantize=quantize),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups, quantize=quantize),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False, quantize=quantize)
        ])
        self.quantize = quantize
        if self.quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
        if self.quantize:
            y = self.conv(x) + self.residual_quantizer(self.shortcut(x))
        else:
            y = self.conv(x) + self.shortcut(x)

        return F.relu(y, inplace=True)


class ResNet50(torch.nn.Module):
    def __init__(self, num_classes, sz = 224, quantize: bool = False):
        super(ResNet50, self).__init__()
        self.quantize = quantize

        self.stem = torch.nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2, quantize=quantize),  # /2    
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)          # /2
        ])

        self.stages = torch.nn.Sequential(*[
            self._make_stage(64, 2048, down_sample=False, quantize=quantize),
        ])

        if self.quantize:
            self.head = torch.nn.Sequential(*[
                quant_nn.QuantAvgPool2d(kernel_size=sz//4, stride=1, padding=0), ### 224/32 = 7    quant_nn.QuantAvgPool2d
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(2048, num_classes)
            ])
        else:
            self.head = torch.nn.Sequential(*[
                torch.nn.AvgPool2d(kernel_size=sz//4, stride=1, padding=0), ### 224/32 = 7
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(2048, num_classes)
            ])

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, quantize):
        layers = [Bottleneck(in_channels, in_channels, down_sample=down_sample, quantize=quantize)]
        layers.append(Bottleneck(in_channels, out_channels, down_sample=down_sample, quantize=quantize))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(self.stem(x))) 

if __name__ == "__main__":
    inputs = torch.rand((1, 3, 224, 224)).cuda()
    model = ResNet50(num_classes=1000, sz = 224).cuda().eval()
    outputs = model(inputs)

    import sys, os
    from __debug import quant_nn, export_onnx_quant   
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
    from lib.fx import replace_residual_add_to_module   
    model = replace_residual_add_to_module(model)

    from lib.quantize import replace_to_quantization_module 
    replace_to_quantization_module(model) 

    onnx_name = "resnet_bottleneck.onnx"  
    export_onnx_quant(model, inputs, onnx_name)
