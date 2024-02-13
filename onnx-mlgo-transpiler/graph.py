from onnx.checker import check_model
from typing import List

class Node:
  def __init__(self, node):
    # TODO: add other necessary params as well
    pass
  
  @classmethod
  def create_node(cls, node):
    if node.op_type == "Gemm":  
      intermediate_ouput = name that is different from all others in inputs and outputs
      node1 = Node('MulMat', node.input[0:2], intermediate_ouput)
      node2 = Node('Add', [intermediate_ouput, node.input[2]], node.output)
      return [node1, node2]
    elif node.op_type == "Relu":
      pass

class Graph:
  def __init__(self, onnx_graph):
    # TODO: make sure that the model is being correctly checked
    check_model(onnx_graph)

    # properties
    self._nodes: List[Node] = []
    self._inputs = [] # type List[Tensor]
    self._outputs = [] # type List[Tensor]

    for node in onnx_graph.graph.node:
      # assert : self._graph is a complete graph with last output as last_output
      # assert : node takes in 1 input, x weights and has 1 output
      node_list = Node.create_node(node)
      
      add node_list to graph
