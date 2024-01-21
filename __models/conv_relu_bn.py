# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, quantize: bool = False):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        # CRB + CRB,  w h remain unchanged 
        if quantize:
            self.double_conv = nn.Sequential(
                quant_nn.QuantConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_channels),
                quant_nn.QuantConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels))
        else:      
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    ## w h to half 
    def __init__(self, in_channels, out_channels, quantize: bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, quantize=quantize))
      
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, bilinear=True, quantize: bool = False):
        super().__init__()
        self._quantize = quantize

        if self._quantize:
            # we not insert q-dq before resize op, because resize not run int8 prec 
            # resize layer compute as fp16 or fp32, so contiguous ops not quant 
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # map to resize
            else:
                self.up = quant_nn.QuantConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:    
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, quantize: bool = False):
        super(OutConv, self).__init__()
        self._quantize = quantize
        if self._quantize:
            self.conv = quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CRB(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, quantize: bool = False):
        super(CRB, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, quantize=quantize))
        self.down1 = (Down(64, 128, quantize=quantize))
        self.up4 = (Up(128, bilinear, quantize=quantize))
        self.outc = (OutConv(128, n_classes, quantize=False))
        self._quantize = quantize
        if self._quantize:
            self.quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        else:
            self.quantizer = torch.nn.Identity()

    def forward(self, x):   #  1, 3, 256, 256
        x1 = self.inc(x)    # -1, 64, 256, 256
        x2 = self.down1(x1) # -1, 128, 128, 128
        x = self.up4(x2)    # -1, 128, 256, 256

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    inputs = torch.rand((1, 3, 640, 640)).cuda()
    model = CRB(n_channels=3, n_classes=2, bilinear=True).cuda().eval() 
    outputs = model(inputs)

    from __debug import summary, quant_nn, export_onnx_quant
    summary(model, input_dims=(3, 640, 640)) 

    model = CRB(n_channels=3, n_classes=2, bilinear=True, quantize=True).cuda().eval() 
    import os
    onnx_name = os.path.basename(__file__).split('.')[0] + ".onnx"  
    export_onnx_quant(model, inputs, onnx_name)
