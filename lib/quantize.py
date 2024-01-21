# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.optim as optim
from torch.cuda import amp
from torch.autograd import Variable
from loguru import logger
from onnxsim import simplify
import onnx

import sys  
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path + "/../pytorch-quantization_v2.1.2")  

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from absl import logging as quant_logging

sys.path.append(cur_path)
from rules import find_quantizer_pairs

class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


def initialize():
    logger.info("{}".format(quant_nn.__path__))

    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantConv2d_WeightOnly.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)
    return calib, quant_nn


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self): 
        try:
          quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        except AttributeError:
          logger.debug("current op only support quan input, {}".format(self.__class__))
          quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)

        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy : Union[str, List[str], Callable], path : str) -> bool:
    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):
        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True  # `path` is layer name, like conv2  relu0
        for item in ignore_policy: # regex str
            if re.match(item, path):
                return True
    return False


def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None) -> None:
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod    

    def recursive_and_replace_module(module, prefix=""):
        #logger.warning("{},{}".format(type(module._modules), len(module._modules))) # <class 'collections.OrderedDict'>,6
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            # logger.info("{},{},{},{}".format(name, submodule, type(submodule), submodule._modules))# conv666, Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)), <class 'torch.nn.modules.conv.Conv2d'>,OrderedDict()

            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    logger.info(f"Quantization: {path} has ignored.")
                    continue
                # logger.info(f"Quantization: {path} not ignored.")
    
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name) 
        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))


def apply_custom_rules_to_quantizer(model : torch.nn.Module, export_onnx_cb : Callable, dims, local_rank=0, lonlp: list = []):
    ## apply rules to graph  eg. pairs = [['conv2','avgpool1']]
    tmp_onnx = "quantization-custom-rules-temp-{}.onnx".format(str(local_rank))

    export_onnx_cb(model, tmp_onnx, dims, run_simpify = False)

    ## based on onnx graph find binding pairs, then modify `_input_quantizer`
    pairs = find_quantizer_pairs(tmp_onnx, lonlp)
    if len(pairs) == 0:
        logger.info("not find, maybe you need check the method of QDQ insert, or not !")
    
    for major, sub in pairs:
        # logger.info(f"Rules: [{sub}, {major}]")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
        # logger.info("{}, {}".format(get_attr_with_path(model, sub), get_attr_with_path(model, major)))
    os.remove(tmp_onnx)


def calibrate_model(model : torch.nn.Module, model_name, data_loader, device, num_calib_batch, calibrator, hist_percentile, out_dir):
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs) 

                    module._amax = module._amax.to(device)
                #print(F"{name:40}: {module}")
        # model.cuda()

    def collect_stats(model, data_loader, device, num_calib_batch):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        logger.debug("enable calib ...")
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        logger.debug("load data ...")
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_calib_batch, desc="Collect stats for calibrating"):
                imgs = Variable(datas[0]).to(device, non_blocking=True).float()
                model(imgs)
                if i >= num_calib_batch:
                    break

        # Disable calibrators
        logger.debug("disable calib ...")
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    if num_calib_batch > 0:
        logger.debug("Calibrating model")
        model.eval()
        collect_stats(model, data_loader, device, num_calib_batch)

        if not calibrator == "histogram":
            logger.debug("max calibration")
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            torch.save({"state_dict": model.state_dict()}, calib_output)
        else:
            for percentile in hist_percentile:
                logger.debug(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save({"state_dict": model.state_dict()}, calib_output)

            for method in ["mse", "entropy"]:
                logger.debug(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save({"state_dict": model.state_dict()}, calib_output)


def finetune(
    model : torch.nn.Module, train_dataloader, per_epoch_callback : Callable = None, preprocess : Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule : Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy : Callable = None
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()

    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue
        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])

    for iepoch in range(nepochs):
        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):
            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
                
            imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)            ################ quant model

                with torch.no_grad():
                    origin_model(imgs) ################ fp32 no-quant model

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


def export_onnx(model, input, file, run_simpify, *args, **kwargs):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    logger.debug("export onnx done !")
    if not run_simpify:
        quant_nn.TensorQuantizer.use_fb_fake_quant = False
        return

    model = onnx.load(file)
    simplify_tmp = file.replace(".onnx", "_with_simplify.onnx")

    onnx.checker.check_model(model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, simplify_tmp)
    logger.debug("simplify onnx done !")

    quant_nn.TensorQuantizer.use_fb_fake_quant = False


def export_onnx_nquant(model, file, size, *args, **kwargs):
    model.eval()
    with torch.no_grad():
        tmp = file.replace(".onnx", "_tmp.onnx")
        device = next(model.parameters()).device
        dummy_inputs = torch.rand(size[0], device=device)

        torch.onnx.export(model, dummy_inputs, tmp, opset_version=13, *args, **kwargs)
        logger.debug("export onnx done !")
        model = onnx.load(tmp)
        os.remove(tmp)

        onnx.checker.check_model(model)
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.checker.check_model(model_simp)
        onnx.save(model_simp, file)
        logger.debug("simplify onnx done !")


def export_onnx_quant(model, file, size, run_simpify:bool = False, input_names=None, output_names=None, dynamic_axes=None):
    device = next(model.parameters()).device
    dummy_inputs = torch.randn(size[0], device=device)
    export_onnx(model, dummy_inputs, file, 
                run_simpify = run_simpify,
                opset_version=13,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes)

