from typing import List
from graph import Graph

def indent_lines(code_list: List[str], space_size: int, tabs: bool = False):
  """Formats a `code_list` into a single string where each line is on a new line.\
Each line has `space_size` spaces before each line. Doesn't add a new line to the\
last line of code.

  Args:
      code_list (List[str]): Each element represents a line of code
      space_size (_type_): Size of spacing before each line
  """
  output = ""
  spacing = ' ' * space_size
  
  for i in range(len(code_list)):
    new_line = '\n' if i != (len(code_list)-1) else ''
    output += f'{spacing}{code_list[i]}{new_line}'
    
  return output

def create_single_layer(var_name: str, mlgo_op: str, input_list):

  # TODO: check whether the number of inputs are valid for the given mlgo_op
  
  input_str = ', '.join(input_list)
  return f'{var_name} := ml.{mlgo_op}(ctx0, {input_str})'

def create_layers(onnx):
  
  mlgo_graph = Graph(onnx)
  output: List[str] = []

  i = 1
  # TODO: extend this for multi-path graphs
  for node in mlgo_graph.graph['nodes']:
    output.append(create_single_layer(f'l{i}', node.op, node.inputs))
    i+=1

  # assert : output is the go language output for the layers
  
#   output = """\
#   // fc1 MLP = Ax + b
#   fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
#   fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)

#   // softmax
#   final := ml.SoftMax(ctx0, fc2)
# """
  
  return output