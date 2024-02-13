import utils

def create_go_boilerplate(file):
  """
  create imports and boilerplate.
  """
  file.write("""\
package main

import (
  "errors"
  "fmt"
  "math/rand"
  "math"
  "time"
  "mlgo/ml"
  "os"
)

"""
  )

def create_model_utils(file):
  """
  create model utils.
  """
  file.write(
"""\
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

def create_hparams_type(file):
  """
  create type for hparams.
  e.g. - mnist_hparams
  """
  # TODO: is this correct? are any variables required here?
  file.write(f"""\
type model_hparams struct{{
  n_input int32;
  n_hidden int32;
  n_classes int32;
}}

"""
  )

def create_model_type(file):
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
type model_struct struct {{

  hparams model_hparams;

{layers}

}}
"""
  )

def create_weight_loading_func(file):
  """
  create function to load model weights
  e.g. - mnist_model_load
  """

  # TODO: abstract layers programmatically.

  file.write(f"""\
func model_load(fname string, model *model_struct) error {{

  file, err := os.Open(fname)
  if err != nil {{
    return err
  }}
  defer file.Close()


  // verify magic
  {{
    magic := readInt(file)
    if magic != 0x67676d6c {{
      return errors.New("invalid model file (bad magic)")
    }}
  }}

  // Read FC1 layer 1
  {{
    n_dims := int32(readInt(file))
    ne_weight := make([]int32, 0)
    for i := int32(0); i < n_dims; i++ {{
      ne_weight = append(ne_weight, int32(readInt(file)))
    }}
    // FC1 dimensions taken from file, eg. 768x500
    model.hparams.n_input = ne_weight[0]
    model.hparams.n_hidden = ne_weight[1]

    model.fc1_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_weight.Data); i++{{
      model.fc1_weight.Data[i] = readFP32(file)
    }}

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_bias = append(ne_bias, int32(readInt(file)))
    }}

    model.fc1_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_bias.Data); i++ {{
      model.fc1_bias.Data[i] = readFP32(file)
    }}
  }}

  // Read Fc2 layer 2
  {{
    n_dims := int32(readInt(file))
    ne_weight := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_weight = append(ne_weight, int32(readInt(file)))
    }}

    // FC1 dimensions taken from file, eg. 10x500
    model.hparams.n_classes = ne_weight[1]

    model.fc2_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_weight.Data); i++{{
      model.fc2_weight.Data[i] = readFP32(file)
    }}

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {{
      ne_bias = append(ne_bias, int32(readInt(file)))
    }}

    model.fc2_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_bias.Data); i++ {{
      model.fc2_bias.Data[i] = readFP32(file)
    }}
    ml.PrintTensor(model.fc2_bias, "model.fc2_bias")

  }}

  return nil
}}

"""
  )

def create_eval_func(file, onnx_model):
  """
  create function to evaluate model
  e.g. - mnist_eval
  """
  
  layers = utils.create_layers(onnx_model)

  # TODO: create input tensor according to onnx
  # TODO: create fc's (layers) according to onnx

  file.write(
  f"""\
func model_eval(model *model_struct, threadCount int, digit []float32) int {{

  ctx0 := &ml.Context{{}}
  graph := ml.Graph{{ThreadsCount: threadCount}}

  input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.n_input))
  copy(input.Data, digit)

{layers}

  // run the computation
  ml.BuildForwardExpand(&graph, final)
  ml.GraphCompute(ctx0, &graph)

  ml.PrintTensor(final, "final tensor")

  maxIndex := 0
  for i := 0; i < 10; i++{{
    if final.Data[i] > final.Data[maxIndex] {{
      maxIndex = i
    }}
  }}
  return maxIndex
}}

"""
  )

def create_main_func(file):
  """
  create inference main function
  e.g. TestMNIST 
  """

  # TODO: decide which variables need to be here based on onnx
  # TODO: decide what to do with ml.SINGLE_THREAD
  # TODO: make sure that the model weights and inputs are accessed properly
  # TODO: make sure that the paths are relative to THIS file as opposed to the shell

  modelFile = "models/ggml-model-f32.bin"
  digitFile = "models/t10k-images.idx3-ubyte"

  file.write(
f"""\
func main() {{
  modelFile := "{modelFile}"
  digitFile := "{digitFile}"

  ml.SINGLE_THREAD = true
  model := new(model_struct)
  if err := model_load(modelFile, model); err != nil {{
    fmt.Println(err)
    return
  }}

  // load a random test digit
  fin, err := os.Open(digitFile)
  if err != nil {{
    fmt.Println(err)
    return
  }}
   // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
  rand.Seed(time.Now().UnixNano())
  fin.Seek(int64(16 + 784 * (rand.Int() % 10000)), 0)
  buf := make([]byte, 784)
  digits := make([]float32, 784)
  if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {{
    fmt.Println(err, count)
    return
  }}

  // render the digit in ASCII
  for row := 0; row < 28; row++{{
    for col := 0; col < 28; col++ {{
      digits[row*28 + col] = float32(buf[row*28 + col])
      var c string
      if buf[row*28 + col] > 230 {{
        c = "*"
      }} else {{
        c = "_"
      }}
      fmt.Printf(c)
    }}
    fmt.Println("")
  }}
  fmt.Println("")

  res := model_eval(model, 1, digits)
  fmt.Println("Predicted digit is ", res)
}}
"""
  )