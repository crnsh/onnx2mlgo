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

type model_hparams struct{
  n_input int32;
  n_hidden int32;
  n_classes int32;
}

type model_struct struct {

  hparams model_hparams;

  fc1_weight *ml.Tensor;
  fc1_bias *ml.Tensor;

  fc2_weight *ml.Tensor;
  fc2_bias *ml.Tensor;  

}
func model_load(fname string, model *model_struct) error {

  file, err := os.Open(fname)
  if err != nil {
    return err
  }
  defer file.Close()


  // verify magic
  {
    magic := readInt(file)
    if magic != 0x67676d6c {
      return errors.New("invalid model file (bad magic)")
    }
  }

  // Read FC1 layer 1
  {
    n_dims := int32(readInt(file))
    ne_weight := make([]int32, 0)
    for i := int32(0); i < n_dims; i++ {
      ne_weight = append(ne_weight, int32(readInt(file)))
    }
    // FC1 dimensions taken from file, eg. 768x500
    model.hparams.n_input = ne_weight[0]
    model.hparams.n_hidden = ne_weight[1]

    model.fc1_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_weight.Data); i++{
      model.fc1_weight.Data[i] = readFP32(file)
    }

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {
      ne_bias = append(ne_bias, int32(readInt(file)))
    }

    model.fc1_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
    for i := 0; i < len(model.fc1_bias.Data); i++ {
      model.fc1_bias.Data[i] = readFP32(file)
    }
  }

  // Read Fc2 layer 2
  {
    n_dims := int32(readInt(file))
    ne_weight := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {
      ne_weight = append(ne_weight, int32(readInt(file)))
    }

    // FC1 dimensions taken from file, eg. 10x500
    model.hparams.n_classes = ne_weight[1]

    model.fc2_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_weight.Data); i++{
      model.fc2_weight.Data[i] = readFP32(file)
    }

    ne_bias := make([]int32, 0)
    for i := 0; i < int(n_dims); i++ {
      ne_bias = append(ne_bias, int32(readInt(file)))
    }

    model.fc2_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_classes))
    for i := 0; i < len(model.fc2_bias.Data); i++ {
      model.fc2_bias.Data[i] = readFP32(file)
    }
    ml.PrintTensor(model.fc2_bias, "model.fc2_bias")

  }

  return nil
}

func model_eval(model *model_struct, threadCount int, digit []float32) int {

  ctx0 := &ml.Context{}
  graph := ml.Graph{ThreadsCount: threadCount}

  input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.n_input))
  copy(input.Data, digit)

  l1 := ml.MulMat(ctx0, input, fc1.weight)
  l2 := ml.Add(ctx0, temp1, fc1.bias)
  l3 := ml.Relu(ctx0, /fc1/Gemm_output_0)
  l4 := ml.MulMat(ctx0, /relu/Relu_output_0, fc2.weight)
  l5 := ml.Add(ctx0, temp3, fc2.bias)

  // run the computation
  ml.BuildForwardExpand(&graph, final)
  ml.GraphCompute(ctx0, &graph)

  ml.PrintTensor(final, "final tensor")

  maxIndex := 0
  for i := 0; i < 10; i++{
    if final.Data[i] > final.Data[maxIndex] {
      maxIndex = i
    }
  }
  return maxIndex
}

func main() {
  modelFile := "models/ggml-model-f32.bin"
  digitFile := "models/t10k-images.idx3-ubyte"

  ml.SINGLE_THREAD = true
  model := new(model_struct)
  if err := model_load(modelFile, model); err != nil {
    fmt.Println(err)
    return
  }

  // load a random test digit
  fin, err := os.Open(digitFile)
  if err != nil {
    fmt.Println(err)
    return
  }
   // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
  rand.Seed(time.Now().UnixNano())
  fin.Seek(int64(16 + 784 * (rand.Int() % 10000)), 0)
  buf := make([]byte, 784)
  digits := make([]float32, 784)
  if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {
    fmt.Println(err, count)
    return
  }

  // render the digit in ASCII
  for row := 0; row < 28; row++{
    for col := 0; col < 28; col++ {
      digits[row*28 + col] = float32(buf[row*28 + col])
      var c string
      if buf[row*28 + col] > 230 {
        c = "*"
      } else {
        c = "_"
      }
      fmt.Printf(c)
    }
    fmt.Println("")
  }
  fmt.Println("")

  res := model_eval(model, 1, digits)
  fmt.Println("Predicted digit is ", res)
}
