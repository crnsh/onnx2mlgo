import utils
from graph import Graph

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

def create_eval_func(file, graph: Graph):
  """
  create function to evaluate model
  e.g. - mnist_eval
  """
  layers = utils.create_layers(graph)
  tensor_initialization = utils.define_and_initialize_tensors(graph)
  output_name = utils.get_assignment_target(layers[-1])
  magic = '0x6d6c676f'
  # TODO: create input tensor according to onnx
  # TODO: create fc's (layers) according to onnx
  # TODO: make sure that the final layer var name matches that of the remaining

  file.write(
  f"""\
func model_eval(fname string, threadCount int) error {{

  file, err := os.Open(fname)
  if err != nil {{
    return err
  }}
  defer file.Close()

  //verify magic
  {{
    magic := readInt(file)
    // {magic} is mlgo in hex
    if magic != {magic} {{
      return errors.New("invalid model file (bad magic)")
    }}
  }}

  ctx0 := &ml.Context{{}}
  graph := ml.Graph{{ThreadsCount: threadCount}}

{utils.indent_lines(tensor_initialization, 2)}

{utils.indent_lines(layers, 2)}

  // run the computation
  ml.BuildForwardExpand(&graph, {output_name})
  ml.GraphCompute(ctx0, &graph)

  ml.PrintTensor({output_name}, "final tensor")

  return nil
}}

"""
  )

def create_main_func(file, model_weights_fname):
  """
  create inference main function
  e.g. TestMNIST 
  """
  # TODO: make sure that the model weights and inputs are accessed properly
  # TODO: make sure that the paths are relative to THIS file as opposed to the shell

  file.write(
f"""\
func main() {{
  
  model_weights_fname := "{model_weights_fname}"
  ml.SINGLE_THREAD = true
  err := model_eval(model_weights_fname, 1)
  if err != nil {{
    fmt.Printf("error : %s\\n", err)
  }}

}}
"""
  )