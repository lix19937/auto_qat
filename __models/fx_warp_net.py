# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
import torch.nn as nn 
import torch.fx  
import torch.nn.functional as F

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
            padding, groups=groups, bias=False)

        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU(inplace=True) if activation else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else torch.nn.Identity()

        self.conv = torch.nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])
  
    def forward(self, x): 
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)

############################################################################################
#
#  TypeError: 'Proxy' object does not support item assignment
#
#  limitation of symbolic tracing with FX. Here is a workaround using @torch.fx.wrap:
#  
#  ref https://pytorch.org/docs/stable/fx.html
#  When symbolic tracing, the below call to my_custom_function will be inserted into
#  the graph rather than tracing it.
############################################################################################
@torch.fx.wrap
def norm(x, down_ratio = 0.5, bev_res = 3.14, roi_offset = 1.0): 
    x[:, 0] = x[:, 0]  *  down_ratio * bev_res                                      
    x[:, :, :-1, :] = x[:, :, :-1, :] * down_ratio + roi_offset
    return x
 
                                                                         
class SimpWarp_M(nn.Module):                                          
    def __init__(self, in_channel, out_channel):                            
        super(SimpWarp_M, self).__init__()                            
        self.bottle = Bottleneck(in_channel, out_channel)

    def forward(self, x):
        x = self.bottle(x)
        x = norm(x)
        return x  
