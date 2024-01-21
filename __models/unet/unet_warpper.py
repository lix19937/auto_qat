# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

# import torch
# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)
# from torch.fx import symbolic_trace
# # Symbolic tracing frontend - captures the semantics of the module
# symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
# # High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)
# print("-----------------------------------")
# print(symbolic_traced.code)

# ref https://github.com/milesial/Pytorch-UNet

import torch

from unet.unet_model import UNet

def load_model(weight, device):
    model = UNet(n_channels=3, n_classes=2, bilinear=True)

    if weight is not None and weight != '':
        state_dict = torch.load(weight, map_location=device)        
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        print(f'Model loaded from {weight}')

    model.to(device=device)

    print(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
            
    model.float().eval()
    return model
    

if __name__ == "__main__":
    inputs = torch.rand((1, 3, 256, 256)).cuda()
    model = load_model(weight='', device="cuda:0").eval() 
    outputs = model(inputs)
    from torchsummary import summary
    summary(model, input_data=(3, 256, 256), depth=3) 
