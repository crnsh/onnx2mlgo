from typing import List
from graph import Graph
import re
from onnx2mlgo import types

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

def create_single_layer(var_name: str, mlgo_op: str, input_list) -> str:
  # TODO: check whether the number of inputs are valid for the given mlgo_op
  input_str = ', '.join(input_list)
  return f'{var_name} := ml.{mlgo_op}(ctx0, {input_str})'

def create_layers(graph: Graph) -> List[str]:
  output: List[str] = []
  # TODO: extend this for multi-path graphs
  for node in graph.nodes:
    output.append(create_single_layer(node.output, node.op, node.inputs))
  # assert : output is a list of go language lines for the defining layers of the nn
  return output

def shape_size(tensor_variant: types.TensorVariant):
  number = re.search(r'\d+', tensor_variant)
  if number:
    return int(number.group())
  else:
    raise ValueError(f'tensor variant ({tensor_variant}) does not indicate shape size')

def define_tensor(var_name: str,
                  tensor_variant: types.TensorVariant,
                  ctx: str,
                  dtype: types.Dtype,
                  shape: List[int]):
  shape_argument = ', '.join([f'uint({dim})' for dim in shape])
  if shape_size(tensor_variant) != len(shape):
    raise ValueError(f'shape size ({len(shape)}) does not match tensor variant ({tensor_variant})')
  return [f'{var_name} = ml.{tensor_variant}({ctx}, ml.{dtype}, {shape_argument})']

def create_for_loop(
  init_statement: types.Statement,
  condition_statement: types.Statement,
  post_statement: types.Statement,
  loop_statements: List[types.Statement]
) -> types.Statement:
  return [f'''\
for {init_statement}; {condition_statement}; {post_statement} {{
  {indent_lines(loop_statements, 2)}
}}
''']

def initialize_tensor(loop_var: str, tensor_var_name: str, filename: str) -> List[str]:
  # TODO: create a codegen library for go
  return create_for_loop(f'{loop_var} := 0', f'{loop_var} < len({tensor_var_name})', f'{loop_var}++',
                         [f'{tensor_var_name}.Data[{loop_var}] = 0.412152'])

def define_and_initialize_tensors(graph: Graph) -> List[str]:
  output = []
  # TODO: there are ml.NewTensor2DWithData type tensors. see if you can integrate them
  for initializer in graph.initializers:
    output += define_tensor(initializer.name, TEST, 'nil', 'TYPE_F32', initializer.shape)
    output += initialize_tensor(initializer)
    output.append('') # new line
  for input in graph.inputs:
    pass
  # assert : output is a list of go language lines for defining and initializing the input and weight tensors
  return output