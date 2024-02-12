from onnx.checker import check_model
from typing import List

class Node:
  def __init__(self, id: str, op: str, *inputs: List[str], output: str, input_first: bool):
    # TODO: add other necessary params as well
    self.id = id
    self.op = op
    self.inputs = inputs
    self.output = output
    self.input_first = input_first

class Graph:
  def __init__(self, onnx_graph):
    # TODO: make sure that the model is being correctly checked
    check_model(onnx_graph)

    # properties
    graph = []

    for node in onnx_graph.graph.node:
      pass


  def in_order(self):
    pass