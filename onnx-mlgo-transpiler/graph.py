from onnx.checker import check_model
from typing import List

class Node:
  def __init__(self, node):
    # TODO: add other necessary params as well
    pass
  
  @classmethod
  def create_node(cls, node, i: int):
    if node.op_type == "Gemm":  
      temp_output = f'temp{i}'
      node1 = Node('MulMat', node.input[0:2], temp_output)
      node2 = Node('Add', [temp_output, node.input[2]], node.output)
      return [node1, node2]
    elif node.op_type == "Relu":
      pass

class Graph:
  def __init__(self, onnx_graph):
    # TODO: make sure that the model is being correctly checked
    check_model(onnx_graph)

    # properties
    self._nodes: List[Node] = []
    self._inputs = onnx_graph.graph.input # type List[Tensor]
    self._outputs = onnx_graph.graph.output # type List[Tensor]

    i = 1
    for onnx_node in onnx_graph.graph.node:
      # assert : self._graph is a complete graph with last output as last_output
      # assert : node takes in 1 input, x weights and has 1 output
      node_list = Node.create_node(onnx_node, i)
      
      for node in node_list:
        self.add_node_to_graph(node)
      
      i += 1
  
  def add_node_to_graph(self, node):
    pass
        
