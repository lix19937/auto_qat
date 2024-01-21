# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import warnings
from typing import Dict, Any
import operator
import torch.fx
from torch.fx import passes

from loguru import logger


def find_candidate_parents(modules: Dict[str, Any], node: torch.fx.Node, parents:list = [], depth:int = 0, conv_cnt:int = 0):
  if not hasattr(node, 'target'):
    return parents, depth, conv_cnt

  if node.target in modules and type(modules[node.target]) is torch.nn.Upsample:
    if depth == 0: 
      return [], 0, -1

  if node.target in modules and type(modules[node.target]) is torch.nn.Conv2d:
    conv_cnt = conv_cnt + 1

  parents.append(node)
  depth = depth + 1 

  if len(node.args) == 0:
    return parents, depth, conv_cnt
  
  node = node.args[0]
  return find_candidate_parents(modules, node, parents, depth, conv_cnt)
 

def find_common_parent(parents_a:list, parents_b:list):
    for n in parents_a:
        for m in parents_b:
            if m.name == n.name:
                return n              
    return None


## shortcut opposite branch has conv
def find_shortcut_branch(fx_model: torch.fx.GraphModule) -> list:
    # add op：
    #     1. x + y, when as Node, target is operator.add    
    #     2. torch.add(x, y)，when as Node, target is torch.add   
    #     3. x.add(y)，when as Node, target is str "add"
    patterns = set([operator.add, torch.add, "add"])

    match_branchs = []
    resize_branchs = []
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:   
        if node.target in patterns:
            parents_a, depth_a, conv_cnt_a = find_candidate_parents(modules, node.args[0]) 
            parents_b, depth_b, conv_cnt_b = find_candidate_parents(modules, node.args[1]) 

            if conv_cnt_a == -1 or conv_cnt_b == -1:
              resize_branchs.append(node) 
              continue

            if depth_a < depth_b:
                par = find_common_parent(parents_a, parents_b)
            else:
                par = find_common_parent(parents_b, parents_a)

            # logger.debug("= {}| {}".format(par, node.args))  
            # logger.debug("= {} {} {}".format(parents_a, depth_a, conv_cnt_a))  
            # logger.debug("= {} {} {}".format(parents_b, depth_b, conv_cnt_b))  

            if par is not None:
              if par.target in modules:
                logger.debug("{}".format(type(modules[par.target])))  
              else:
                logger.debug("{}".format(par.target))  

            if par is not None:
                shortcut_side = 0 if depth_a < depth_b else 1
                if (shortcut_side == 0 and conv_cnt_a < conv_cnt_b) or (shortcut_side == 1 and conv_cnt_a > conv_cnt_b):
                  t = [node, shortcut_side] 
                  match_branchs.append(t) 

                logger.debug("{} {} {}".format(shortcut_side, conv_cnt_a, conv_cnt_b)) 
            else:
                logger.warning("cur add no common parent, donot quant")   
  
    return match_branchs, resize_branchs


def replace_residual_add_to_module(model: torch.nn.Module)-> torch.nn.Module:
  fx_model = torch.fx.symbolic_trace(model)  # if node.target in modules and type(modules[node.target]) is torch.nn.Conv2d:
  # fx_model.graph.print_tabular()

  # modules = dict(fx_model.named_modules())
  # for node in fx_model.graph.nodes:   
  #   if node.target in modules:
  #     print(node.name, type(modules[node.target]))
  #   else:
  #     print("-----------", node.name)  
  # exit(0)
  add_pairs, resize_pairs = find_shortcut_branch(fx_model)  

  for i in range(len(add_pairs)):
    fx_model.add_submodule("residual_add" + str(i), torch.nn.MSELoss())

    target = add_pairs[i][0]
    logger.info("{}{}{}".format(target, target.args, target.name)) 

    with fx_model.graph.inserting_before(target):
        with warnings.catch_warnings(record=True) as w:
            args = target.args[::-1] if add_pairs[i][1] == 1 else target.args
            dropout = fx_model.graph.call_module(module_name="residual_add" + str(i), args=args)

    target.replace_all_uses_with(dropout)
    fx_model.graph.erase_node(target)

  for i in range(len(resize_pairs)):
    fx_model.add_submodule("residual_add_resize" + str(i), torch.nn.L1Loss())

    target = resize_pairs[i]
    logger.info("{}{}{}".format(target, target.args, target.name)) 

    with fx_model.graph.inserting_before(target):
        with warnings.catch_warnings(record=True) as w:
            args = target.args
            dropout = fx_model.graph.call_module(module_name="residual_add_resize" + str(i), args=args)

    target.replace_all_uses_with(dropout)
    fx_model.graph.erase_node(target)

  fx_model.recompile()
  model = torch.fx.GraphModule(fx_model, fx_model.graph)

  ## pip3 install pydot
  # draw = passes.graph_drawer.FxGraphDrawer(fx_model, 'resnet50')
  # with open("resnet50.svg", "wb") as f:
  #     f.write(draw.get_main_dot_graph().create_svg())
  
  return model
