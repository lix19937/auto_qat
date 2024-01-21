# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-04-29 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-04-29 11:09:48
#  **************************************************************/

import torch

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        print("weight.shape: ", self.conv.weight.shape)

    def forward(self, x):   
        x = self.conv(x)   
        return x


if __name__ == "__main__":
    inputs = torch.rand((1, 3, 640, 640)).cuda()
    model = Conv(in_channels=3, out_channels=2).cuda().eval() 
    outputs = model(inputs)

    from __debug import summary
    summary(model, input_dims=(3, 640, 640)) 
