from onnx.checker import check_model
from typing import List, Set
import utils

EXIT_ON_UNSUPPORTED_OP = False

class Node:
  unsupported_ops: Set[str] = set()
  temp_cnt = 0
  def __init__(self, mlgo_op: str, inputs: List[str], output: str):
    self._op = mlgo_op
    self._inputs = inputs
    self._output = output
    self._input_first = True

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
    return self._output

  @classmethod
  def create_node(cls, onnx_node) -> List:
    node_inputs = utils.sanitize_list(onnx_node.input)
    node_output = utils.sanitize_string(onnx_node.output[0])
    op = onnx_node.op_type
    if len(onnx_node.output) > 1:
      raise NotImplementedError(f'onnx nodes with multiple outputs are currently not supported')
    if op == "Gemm":
      temp_output = f'temp{Node.temp_cnt}'
      Node.temp_cnt+=1
      node1 = Node('MulMat', [node_inputs[1], node_inputs[0]], temp_output)
      node2 = Node('Add', [temp_output, node_inputs[2]], node_output)
      return [node1, node2]
    elif op == "Relu":
      node = Node('Relu', node_inputs, node_output)
      return [node]
    else:
      Node.unsupported_ops.add(op) if op not in Node.unsupported_ops else None
      if EXIT_ON_UNSUPPORTED_OP:
        raise NotImplementedError(f'{onnx_node.op_type} has not been implemented yet.')
      return []

class Input:
  def __init__(self, onnx_input):
    self._name = utils.sanitize_string(onnx_input.name)
    self._type = onnx_input.type

  @property
  def name(self):
    return self._name

  @property
  def type(self):
    return self._type

  def get_shape(self):
    shape = str(self.type.tensor_type.shape.dim)
    input_dims = [int(s) for s in shape.split() if s.isdigit()]
    return input_dims

class Initializer:
  def __init__(self, onnx_initializer):
    self._dims = onnx_initializer.dims
    self._data_type = onnx_initializer.data_type
    self._name = utils.sanitize_string(onnx_initializer.name)
    self._raw_data = onnx_initializer.raw_data

  @property
  def dims(self):
    return self._dims

  @property
  def data_type(self):
    return self._data_type

  @property
  def name(self):
    return self._name

  @property
  def raw_data(self):
    return self._raw_data

class Graph:
  def __init__(self, onnx_graph):
    # TODO: sanitize all strings in input, output and initializer. might need to create separate classes for input, initializers and output
    check_model(onnx_graph)
    self._nodes: List[Node] = []
    self._inputs: List[Input] = [] # type List[Tensor]
    self._initializers: List[Initializer] = [] # type List[Tensor]
    self._outputs = onnx_graph.graph.output # type List[Tensor]
    for onnx_node in onnx_graph.graph.node:
      # assert : self._graph is a complete graph with last output as last_output
      # assert : node takes in 1 input, x weights and has 1 output
      node_list = Node.create_node(onnx_node)
      for node in node_list:
        self.add_node(node)
    for onnx_input in onnx_graph.graph.input:
      self._inputs.append(Input(onnx_input))
    for onnx_initializer in onnx_graph.graph.initializer:
      self._initializers.append(Initializer(onnx_initializer))
    if (len(Node.unsupported_ops) > 0):
      raise NotImplementedError(f'Some operations {Node.unsupported_ops} are not currently supported.')

  @property
  def nodes(self):
    return self._nodes

  @property
  def inputs(self):
    return self._inputs

  @property
  def initializers(self):
    return self._initializers

  @property
  def outputs(self):
    return self._outputs

  def add_node(self, node):
    self._nodes.append(node)
        
