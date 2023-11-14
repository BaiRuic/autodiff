from typing import List, Dict
from collections import defaultdict
import copy

def compute_gradient_of_variables(output_tensor:"Tensor", out_grad:"Tensor"):
    node_to_output_grads_list:Dict["Tensor", List["Tensor"]] = defaultdict(list)
    node_to_output_grads_list[output_tensor] = [out_grad]

    reversr_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reversr_topo_order:
        # 先计算得到当前节点的梯度
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if node.is_leaf:
            continue
        # 然后再把待回传的梯度赋给当前节点的输入节点
        in_nodes_grads = node.op.gradient_as_tuple(node.grad, node)

        for in_node, grad in zip(node.inputs, in_nodes_grads):
            node_to_output_grads_list[in_node].append(grad)

def find_topo_sort(node_list:List["Tensor"]):
    topo_order = []
    visited = set()

    for cur_node in node_list:
        topo_sort_dfs(cur_node, topo_order, visited)

    return topo_order

def topo_sort_dfs(cur_node, res, visited):
    if cur_node in visited:
        return
    
    for in_node in cur_node.inputs:
        topo_sort_dfs(in_node, res, visited)

    res.append(cur_node)
    visited.add(cur_node)

def sum_node_list(node_list:List["Tensor"]):
    # 这里需要深拷贝，因为需要创建扩展计算图
    if len(node_list) == 1:
        return copy.deepcopy(node_list[0])
    
    res = node_list[0]
    for node in node_list[1:]:
        res = res + node
    return res