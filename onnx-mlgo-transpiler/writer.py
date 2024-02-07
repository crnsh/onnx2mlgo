def create_model_utils(file):
  """
  create model utils.
  """
  file.write(
"""\
package utils

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

def create_go_boilerplate(file):
  """
  create imports and boilerplate.
  """
  # TODO: remove redundant imports
  file.write("""\
package main

import (
  "errors"
  "utils"
  "fmt"
  "math"
  "mlgo/ml"
  "os"
)

"""
  )

def create_hparams_type(file, model_name: str):
  """
  create type for hparams.
  e.g. - mnist_hparams
  """
  # TODO: is this correct? are any variables required here?
  file.write(f"""\
type {model_name}_hparams struct{{
  n_input int32;
  n_hidden int32;
  n_classes int32;
}}

"""
  )

def create_model_type(file, model_name: str):
  """
  create model type
  e.g. - mnist_model
  """
  # TODO: abstract the logic from the string manipulation output
  # TODO: change these depending on the model weights and layers
  layers = """\
  fc1_weight *ml.Tensor;
  fc1_bias *ml.Tensor;

  fc2_weight *ml.Tensor;
  fc2_bias *ml.Tensor;\
  """

  file.write(f"""\
type {model_name}_model struct {{

  hparams {model_name}_hparams;

{layers}

}}
"""
  )

def create_weight_loading_func(file, model_name: str):
  """
  create function to load model weights
  e.g. - mnist_model_load
  """

  # TODO: abstract layers programmatically.

  file.write(f"""\
func {model_name}_model_load(fname string, model *{model_name}_model) error {{

  file, err := os.Open(fname)
  if err != nil {{
    return err
  }}
  defer file.Close()


  // verify magic
  {{
    magic := utils.readInt(file)
    if magic != 0x67676d6c {{
      return errors.New("invalid model file (bad magic)")
    }}
  }}

  // Read FC1 layer 1
  {{
    n_dims := int32(utils.readInt(file))
    ne_weight := make([]int32, 0)
    for i := int32(0); i < n_dims; i++ {{
      ne_weight = append(ne_weight, int32(utils.readInt(file)))
    }}
    // FC1 dimensions taken from file, eg. 768x500
    model.hparams.n_input = ne_weight[0]
    model.hparams.n_hidden = ne_weight[1]

    model.fc1_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_weight.Data); i++{{
      model.fc1_weight.Data[i] = utils.readFP32(file)
    }}

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_bias = append(ne_bias, int32(utils.readInt(file)))
    }}

    model.fc1_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_bias.Data); i++ {{
      model.fc1_bias.Data[i] = utils.readFP32(file)
    }}
  }}

  // Read Fc2 layer 2
  {{
    n_dims := int32(utils.readInt(file))
    ne_weight := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_weight = append(ne_weight, int32(utils.readInt(file)))
    }}

    // FC1 dimensions taken from file, eg. 10x500
    model.hparams.n_classes = ne_weight[1]

    model.fc2_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_weight.Data); i++{{
      model.fc2_weight.Data[i] = utils.readFP32(file)
    }}

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_bias = append(ne_bias, int32(utils.readInt(file)))
    }}

    model.fc2_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_bias.Data); i++ {{
      model.fc2_bias.Data[i] = utils.readFP32(file)
    }}
    ml.printTensor(model.fc2_bias, "model.fc2_bias")

  }}

  return nil
}}

"""
  )

def create_eval_func(file):
  """
  create function to evaluate model
  e.g. - mnist_eval
  """
  pass

def create_main_func(file):
  """
  create inference main function
  e.g. TestMNIST 
  """
  pass