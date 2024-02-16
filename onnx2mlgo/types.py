from typing import List, Literal

Line = List[str] # A single line of code

Dtype = Literal[
  'TYPE_F32',
  'TYPE_F16',
  'TYPE_Q4_0',
  'TYPE_Q4_1',
  'TYPE_I8',
  'TYPE_I16',
  'TYPE_I32',
  'TYPE_COUNT'
]

TensorVariant = Literal[
  'NewTensor1D',
  'NewTensor2D',
  'NewTensor3D',
  'NewTensor4D',
]