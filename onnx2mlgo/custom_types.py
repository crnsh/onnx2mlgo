from typing import List, Literal

# TODO: Statement should most probably be a str instead of List[str]
# TODO: Statement should consist of a List[Line], where Line = str

Statement = List[str]  # A single statement of code (not necessarily a single line)

Dtype = Literal[
    "TYPE_F32",
    "TYPE_F16",
    "TYPE_Q4_0",
    "TYPE_Q4_1",
    "TYPE_I8",
    "TYPE_I16",
    "TYPE_I32",
    "TYPE_COUNT",
]

TensorVariant = Literal[
    "NewTensor1D",
    "NewTensor2D",
    "NewTensor3D",
    "NewTensor4D",
]
