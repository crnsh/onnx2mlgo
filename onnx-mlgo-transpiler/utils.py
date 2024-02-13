from typing import List
from graph import Graph

def create_graph(onnx):

  # assert : output is a graph where each node a valid mlgo operation.

  return output 

def create_single_layer(var_name: str, mlgo_op: str, input_list):

  # TODO: check whether the number of inputs are valid for the given mlgo_op
  
  input_str = ', '.join(input_list)
  return f'{var_name} := ml.{mlgo_op}(ctx0, {input_str})'

def create_layers(onnx):
  
  mlgo_graph = Graph(onnx)
  output = ""

  i = 1
  # TODO: extend this for multi-path graphs
  for node in mlgo_graph.in_order:
    
    op_input: str = GET_OP_INPUT
    op_weights: List[str] = GET_OP_WEIGHTS

    input_list = None

    if (node.input_first):
      input_list = [op_input]
      input_list += op_weights
    else:
      input_list = op_weights
      input_list += [op_input]

    output.append(create_single_layer(f'l{i}', node.mlgo_op, input_list))

  # assert : output is the go language output for the layers
  
  output = """\
  // fc1 MLP = Ax + b
  fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
  fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)

  // softmax
  final := ml.SoftMax(ctx0, fc2)  
"""
  
  return output