# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

# ref https://github.com/WongKinYiu/yolov7

import torch

from models.yolo import Model
from models.common import Conv
from utils.google_utils import attempt_download

def load_model(weight, device) -> Model:
    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is torch.nn.Upsample:
            m.recompute_scale_factor = None  # pytorch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            
    model.float().eval()

    with torch.no_grad():
        model.fuse()
    return model

# def export_onnx(model : Model, file, size=640, dynamic_batch=False):
#     device = next(model.parameters()).device
#     model.float()

#     dummy = torch.zeros(1, 3, size, size, device=device)
#     model.model[-1].concat = True
#     grid_old_func = model.model[-1]._make_grid
#     model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())

#     quantize.export_onnx(model, dummy, file, opset_version=13, 
#         input_names=["images"], output_names=["outputs"], 
#         dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
#     )
#     model.model[-1].concat = False
#     model.model[-1]._make_grid = grid_old_func

# pip uninstall torchsummary
# pip install torch-summary==1.4.4


if __name__ == "__main__":
    inputs = torch.rand((1, 3, 640, 640)).cuda()
    model = load_model(weight="./yolov7.pt", device="cuda:0").eval() 
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        print(len(outputs)) 
    # https://github.com/TylerYep/torchinfo
    from torchsummary import summary
    summary(model, input_data=(3, 640, 640), depth=3) 
    