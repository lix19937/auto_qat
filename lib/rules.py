# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import onnx
from loguru import logger  

# find first node by name_str match input
def find_with_input_node(model, name):
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            return node

# find nodes by name_str match input
def find_all_with_input_node(model, name):
    all = []
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all

# find node by name_str match output
def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node
    return None

# find parent node 
def find_with_no_change_parent_node(model, node):
    parent = find_with_output_node(model, node.input[0])
    if parent is not None:
        # ref https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qdq-limitations
        if parent.op_type in ["Concat", "MaxPool"]: # be careful: "AveragePool" not commute layer
            return find_with_no_change_parent_node(model, parent)
    return parent


def find_common_parents(model, input, parents:list, depth:int):
    parent = find_with_output_node(model, input)
    if parent is not None:
        if len(parent.output) >= 1:
            input_0 = parent.input[0]

            parents.append(parent.name)
            return find_common_parents(model, input_0, parents, depth+1)
    else:
        return parents, depth


def find_common_parent(parents_a:list, parents_b:list):
    for n in parents_a:
        for m in parents_b:
            if m == n:
                return n              
    return None


## Assume 2 branchs(from one parent) merge to Add
# Add has 2 inputs(Mark as A), search these inputs`s parent nodes, 
# if one of parent nodes has 2 output, stop search of this input link, but another input link continue
# if one of parent nodes only 1 output, continue search 
# if parent node only 2 outputs(Mark as B), and EQ A
# if one of them(Bi) is Identity to Aj, so Bi is marked as need_quant_in
# if none of of them(Bi) is Identity to Aj, we just select any one marked as need_quant_in
def find_shortcut_branch(model) -> list:
    match_branchs = []
    for node in model.graph.node:   
        if node.op_type == "Add":
            depth_a, depth_b = 0, 0
            parents_a, parents_b = [], []
            parents_a, depth_a = find_common_parents(model, node.input[0], parents_a, depth_a) 
            parents_b, depth_b = find_common_parents(model, node.input[1], parents_b, depth_b) 

            if depth_a < depth_b:
                pa = find_common_parent(parents_a, parents_b)
            else:
                pa = find_common_parent(parents_b, parents_a)

            logger.debug("= {}".format(pa))  

            if pa is not None:
                t = [pa, 0 if depth_a < depth_b else 1]
                match_branchs.append(t) 
            else:
                logger.warning("cur add no common parent")   
  
    return match_branchs


def find_quantizelinear_conv(model, qnode):
    dq   = find_with_input_node(model, qnode.output[0])
    conv = find_with_input_node(model, dq.output[0])

    if conv.op_type == "Conv": return conv, 1
    if conv.op_type == "AveragePool": return conv, 0
    if conv.op_type == "Pad":
      pad = conv
      avg = find_with_input_node(model, pad.output[0])
      if avg.op_type == "AveragePool":
        return avg, 0
      else:
        logger.exception("not support, {} ".format(avg));exit(0)
    logger.exception("not support, {} ".format(conv));exit(0)


def find_quantize_conv_name(model, conv, idx, lonlp_map):
    if conv.op_type != "Conv": 
      lonlp_list = lonlp_map[0]
      if len(lonlp_list) > 0:
        lonlp_idx = lonlp_map[1]
        nstr = lonlp_list[lonlp_idx]
        lonlp_map[1] = lonlp_map[1] + 1
        return nstr
      else:
        logger.error("some error case")
        return None 

    weight_qname = conv.input[idx]
    dq = find_with_output_node(model, weight_qname)
    if len(dq.input) > 0:
      q  = find_with_output_node(model, dq.input[0])
      nstr = ".".join(q.input[0].split(".")[:-1]) # resBlock1.conv5.weight --> resBlock1.conv5
      return nstr
  
    logger.exception("dq has no input\n{}\nweight_qname:{}".format(dq, weight_qname)) 
    return None


def find_quantizer_pairs(onnx_file, lonlp: list = []) -> list:
    # logger.info("load model:{}".format(onnx_file))
    model = onnx.load(onnx_file)

    # match_shorts = find_shortcut_branch(model)

    match_pairs = []
    lonlp_map = [lonlp, 0]
    for node in model.graph.node:   
        if node.op_type == "Concat":
            # concat -> q  maybe concat has some subnodes
            qnodes = find_all_with_input_node(model, node.output[0])
            major = None
            for qnode in qnodes:
                if qnode.op_type != "QuantizeLinear":
                    continue
                
                # q-dq-conv or q-dq-pad-avgpool or q-dq-avgpool
                conv, idx = find_quantizelinear_conv(model, qnode)
                if major is None:
                    major = find_quantize_conv_name(model, conv, idx, lonlp_map)
                else:
                    ext = find_quantize_conv_name(model, conv, idx, lonlp_map)
                    if ext is not None:
                        match_pairs.append([major, ext])
						  
                for subnode in model.graph.node:
                    if len(subnode.input) > 0 and subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        subconv, idx = find_quantizelinear_conv(model, subnode)
                        ext = find_quantize_conv_name(model, subconv, idx, lonlp_map)
                        if ext is not None:
                            match_pairs.append([major, ext])
						
        elif node.op_type == "MaxPool": #or node.op_type == "Resize": 
            qnode = find_with_input_node(model, node.output[0])
            if not (qnode and qnode.op_type == "QuantizeLinear"):
                continue

            major, idx = find_quantizelinear_conv(model, qnode)
            major = find_quantize_conv_name(model, major, idx, lonlp_map)
            same_input_nodes = find_all_with_input_node(model, node.input[0])

            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear":
                    subconv, idx = find_quantizelinear_conv(model, same_input_node)
                    ext = find_quantize_conv_name(model, subconv, idx, lonlp_map)
                    if major is not None and ext is not None:
                        match_pairs.append([major, ext])
    
    return match_pairs

#
# Resize is support QAT in tensort, And It is a commute OP for Q nodes.  

# Pooling must have the same input and output dynamic range.

# Concat should propagate dynamic range from outputs to inputs to avoid
# Re-quantization during the concatenation
# For layers such as average-pooling, It support QAT but not a commute OP, Just place a QDQ before and after this OP and see the graph,
