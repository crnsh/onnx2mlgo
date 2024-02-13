from onnx.checker import check_model
from typing import List

class Node:
  def __init__(self, mlgo_op: str, inputs: List[str], output: str):
    # TODO: add other necessary params as well
    self._op = mlgo_op
    self._inputs = inputs    
    self._output = output
    self._input_first = True

  # TODO: figure out this whole _input_first thing
  @property
  def input_first(self):
    return self._input_first

  @property
  def op(self):
    return self._op

  @property
  def inputs(self):
    return self._inputs

  @property
  def output(self):
    return self._outputs

  @classmethod
  def create_node(cls, onnx_node, i: int):
    if onnx_node.op_type == "Gemm":  
      temp_output = f'temp{i}'
      node1 = Node('MulMat', onnx_node.input[0:2], temp_output)
      node2 = Node('Add', [temp_output, onnx_node.input[2]], onnx_node.output)
      return [node1, node2]
    elif onnx_node.op_type == "Relu":
      node = Node('Relu', onnx_node.input, onnx_node.output)
      return node

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
      print(onnx_node)
      node_list = Node.create_node(onnx_node, i)
      
      for node in node_list:
        self.add_node_to_graph(node)
      
      i += 1

  @property
  def graph(self):
    return {'graph': self._graph, 'inputs': self._inputs, 'outputs': self._outputs}
  
  def add_node_to_graph(self, node):
    pass
        
