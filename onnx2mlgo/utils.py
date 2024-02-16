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

def create_single_layer(output_var: str, mlgo_op: str, input_list) -> str:

  # TODO: check whether the number of inputs are valid for the given mlgo_op
  
  input_str = ', '.join(input_list)
  return f'{output_var} := ml.{mlgo_op}(ctx0, {input_str})'

def create_layers(onnx) -> List[str]:
  
  mlgo_graph = Graph(onnx)
  output: List[str] = []

  # TODO: extend this for multi-path graphs
  for node in mlgo_graph.graph['nodes']:
    output.append(create_single_layer(node.output, node.op, node.inputs))

  # assert : output is a list of go language lines for the defining layers of the nn
  
  return output

def initialize_tensors(onnx) -> List[str]:

  # assert : output is a list of go language lines for defining and initializing the input and weight tensors

  return output