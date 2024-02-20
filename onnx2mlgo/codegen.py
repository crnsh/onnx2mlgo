import utils
from graph import Graph
from functools import reduce

def create_go_boilerplate_and_model_utils(file):
  """
  create imports and boilerplate.
  """

  file.write("""\
package main

import (
  "math"
  "mlgo/ml"
  "os"
  "fmt"
  "errors"
)

func readFP32(file *os.File) float32 {
  buf := make([]byte, 4)
  if count, err := file.Read(buf); err != nil || count != 4 {
    return 0.0
  }
  bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
  return math.Float32frombits(bits)
}

func readInt(file *os.File) uint32 {
  buf := make([]byte, 4)
  if count, err := file.Read(buf); err != nil || count != 4 {
    return 0
  }
  return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

"""
  )

def create_eval_func(file, graph: Graph, input_data_var):
  """
  create function to evaluate model
  e.g. - mnist_eval
  """
  magic = '0x6d6c676f'
  tensor_initialization = utils.define_and_initialize_tensors(graph, input_data_var)
  layers = utils.create_layers(graph)
  output_name = utils.get_assignment_target(layers[-1])
  # TODO: create input tensor according to onnx
  # TODO: create fc's (layers) according to onnx
  # TODO: make sure that the final layer var name matches that of the remaining
  # TODO: add dtype to inputData

  file.write(
  f"""\
func model_eval(fname string, threadCount int, {input_data_var} []float32) *ml.Tensor {{

  file, err := os.Open(fname)
  if err != nil {{
    fmt.Println(err)
    os.Exit(1)
  }}
  defer file.Close()

  //verify magic
  {{
    magic := readInt(file)
    // {magic} is mlgo in hex
    if magic != {magic} {{
      fmt.Println(errors.New("invalid model file (bad magic)"))
      os.Exit(1)
    }}
  }}

  ctx0 := &ml.Context{{}}
  graph := ml.Graph{{ThreadsCount: threadCount}}

{utils.indent_lines(tensor_initialization, 2)}

{utils.indent_lines(layers, 2)}

  // run the computation
  ml.BuildForwardExpand(&graph, {output_name})
  ml.GraphCompute(ctx0, &graph)

  return {output_name}
}}

"""
  )

def create_main_func(file, model_weights_fname, input_data_var, input_data_shape):
  """
  create inference main function
  e.g. TestMNIST 
  """
  # TODO: make sure that the model weights and inputs are accessed properly
  # TODO: make sure that the paths are relative to THIS file as opposed to the shell
  # TODO: make []float32 not hardcoded

  input_data_shape_args = reduce(lambda x,y: x*y, input_data_shape)

  file.write(
f"""\
func main() {{
  
  model_weights_fname := "{model_weights_fname}"
  ml.SINGLE_THREAD = true

  {input_data_var} := make([]float32, {input_data_shape_args})
  output_tensor := model_eval(model_weights_fname, 1, {input_data_var})

  ml.PrintTensor(output_tensor, "final tensor")

}}
"""
  )