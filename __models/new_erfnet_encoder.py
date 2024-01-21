
from torch import nn
import torch.nn.functional as F
import torch
from __debug import summary, quant_nn, export_onnx_quant

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, quantize=False):
        super().__init__()
        self.quantize = quantize
        if quantize:
            _Conv2d = quant_nn.QuantConv2d
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        else:
            _Conv2d = nn.Conv2d
            
        self.conv3x1_1 = _Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=False)
        self.conv1x3_1 = _Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = _Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0),
                                   bias=False, dilation=(dilated, 1))
        self.conv1x3_2 = _Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated),
                                   bias=False, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)
        if self.quantize:
            return F.relu(output + self.residual_quantizer(input))
        else:
            return F.relu(output + input)

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, quantize=False):
        super().__init__()
        self.quantize = quantize
        if quantize:
            _Conv2d = quant_nn.QuantConv2d
            _MaxPool2d = nn.MaxPool2d
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        else:
            _Conv2d = nn.Conv2d
            _MaxPool2d = nn.MaxPool2d
        self.conv = _Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=False)
        self.pool = _MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        if self.quantize:
            output = self.residual_quantizer(torch.cat([self.conv(input), self.pool(input)], 1))
        else:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
            
        output = self.bn(output)
        if self.quantize:
            return self.residual_quantizer(F.relu(output))
        else:
            return F.relu(output)

class DownsamplerBlockLJW(nn.Module):
    def __init__(self, ninput, noutput, quantize=False):
        super().__init__()
        self.quantize = quantize
        if quantize:
            _Conv2d = quant_nn.QuantConv2d #######
            _MaxPool2d = quant_nn.QuantMaxPool2d
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
            self.residual_quantizer2 = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

        else:
            _Conv2d = nn.Conv2d
            _MaxPool2d = nn.MaxPool2d
        self.conv = _Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=False)
        self.pool = _MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        if self.quantize:
            # input = self.residual_quantizer2(input)
            output = self.residual_quantizer(torch.cat([self.conv(input), self.pool(input)], 1))
        else:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
            
        output = self.bn(output)
        return F.relu(output)

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, quantize=False):
        super().__init__()
        self.quantize = quantize
        if quantize:
            _ConvTranspose2d = quant_nn.QuantConvTranspose2d
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        else:
            _ConvTranspose2d = nn.ConvTranspose2d

        self.conv = _ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        if self.quantize:
            output = self.residual_quantizer(self.conv(input))
        else:
            output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class NewERFNetEncoder(nn.Module):
    def __init__(self,
                dropout_1=0.03,
                dropout_2=0.1,
                quantize=False
                ):
        super().__init__()
        self.quantize = quantize
        if quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        self.initial_block = DownsamplerBlockLJW(3, 16, quantize=quantize)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlockLJW(16, 64, quantize=quantize))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, dropout_1, 1, quantize=quantize))

        self.layers_1 = nn.ModuleList()
        self.layers.append(DownsamplerBlockLJW(64, 128, quantize=quantize))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, dropout_2, 2, quantize=quantize))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 4, quantize=quantize))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 8, quantize=quantize))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 16, quantize=quantize))

        #add Hanson
        self.layers.append(UpsamplerBlock(128, 64, quantize=quantize))
        self.layers.append(non_bottleneck_1d(64, 0, 1, quantize=quantize))
        self.layers.append(non_bottleneck_1d(64, 0, 1, quantize=quantize))


    def forward(self, input):
        B, N, C, imH, imW = input.shape
        input = input.view(B * N, C, imH, imW)
        output = self.initial_block(input)

        for i, layer in enumerate(self.layers):
            if i==6 and self.quantize: 
                output = self.residual_quantizer(output)

            output = layer(output)

        return [output], [output], [output]


if __name__ == "__main__":
    inputs = torch.rand((2,1,3,256,960)).cuda()


    model = NewERFNetEncoder(quantize=True).cuda().eval() 
    import os
    onnx_name = os.path.basename(__file__).split('.')[0] + ".onnx"  
    export_onnx_quant(model, inputs, onnx_name)

    # import sys
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
    # from lib.fx import replace_residual_add_to_module   
    # model = replace_residual_add_to_module(model)

    # from lib.quantize import replace_to_quantization_module 
    # replace_to_quantization_module(model) 

    # onnx_name = os.path.basename(__file__).split('.')[0] + "_v2.onnx"  
    # export_onnx_quant(model, inputs, onnx_name)