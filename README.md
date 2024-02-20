<div align="center">
  <img src="https://github.com/crnsh/onnx2mlgo/assets/79533543/a7c3c0e1-277d-4079-b827-7ae2fb566493" width=500>
</div>
<p align="center">
  <img src="https://img.shields.io/github/commit-activity/t/crnsh/onnx2mlgo">
  <img src="https://img.shields.io/badge/Working-green">
<p/>
<hr>

**onnx2mlgo** is an ONNX to MLGO transpiler.

Features:
* Single CLI command to output Go files
* Compatibility Checker shows all missing operators
* Code generator

Currently this transpiler only transpiles a limited number of models. The CLI tells you which operations are missing and need to be implemented. Only ONNX models with initialized weights are supported.

You will have to manually import input data in the generated Go file.

If you find a bug, please [let me know](https://github.com/crnsh/onnx2mlgo/issues)!

## Installation
```bash
git clone git@github.com:crnsh/onnx2mlgo.git
cd onnx2mlgo
git submodule update --init --recursive
python3 -m pip install -r requirements.txt
```

For subsequent updates to the `mlgo` library, use the following command.

```bash
git submodule update --recursive --remote
```

## Examples
Make sure you have Go installed!

This example shows how an MNIST ONNX model is transpiled to MLGO.

```bash
python3 onnx2mlgo/transpiler.py tests/mnist_fc.onnx -o mlgo/dist
cd mlgo/dist
go run model.go
```

To check whether the MNIST model works as expected, replace the `main` function of the transpiled `model.go` with the following `main` function
```go
func main() {

	model_weights_fname := "models/model-weights-f32.bin"
	ml.SINGLE_THREAD = true

	inputData := make([]float32, 784)

	// load a random test digit
	digitFile := "models/t10k-images.idx3-ubyte"
	fin, err := os.Open(digitFile)
	if err != nil {
		fmt.Println(err)
		return
	}
	// Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
	rand.Seed(time.Now().UnixNano())
	fin.Seek(int64(16+784*(rand.Int()%10000)), 0)
	buf := make([]byte, 784)
	if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {
		fmt.Println(err, count)
		return
	}

	// render the digit in ASCII
	for row := 0; row < 28; row++ {
		for col := 0; col < 28; col++ {
			inputData[row*28+col] = float32(buf[row*28+col])
			var c string
			if buf[row*28+col] > 230 {
				c = "*"
			} else {
				c = "_"
			}
			fmt.Printf(c)
		}
		fmt.Println("")
	}
	fmt.Println("")

	output_tensor := model_eval(model_weights_fname, 1, inputData)

	ml.PrintTensor(output_tensor, "final tensor")

	maxIndex := 0
	for i := 0; i < 10; i++ {
		if output_tensor.Data[i] > output_tensor.Data[maxIndex] {
			maxIndex = i
		}
	}

	fmt.Println("Predicted digit is ", maxIndex)
}
```

Paste the MNIST input from the `mlgo` submodule to `dist/models`.
```bash
cp ../examples/mnist/models/mnist/t10k-images.idx3-ubyte models/
```

Run the `model.go` file
```bash
go run model.go
```

The output should be the following.
```
____________________________
____________________________
____________________________
____________________________
____________________________
________________________**__
_____________*______******__
____________*************___
____________**********______
____________**______________
___________***______________
___________***______________
__________******____________
_________********___________
__________*******___________
_______________**___________
_______________**___________
_______________**___________
_______________**___________
________****___**___________
________****__***___________
________***__***____________
________*******_____________
_________*****______________
__________**________________
____________________________
____________________________
____________________________



=== [ final tensor | FP32 | 10:1:1 ] ===

 0 x 10 ...     -4070.744       -2108.515       -4699.833       -1760.301       -2717.388       2783.126        -4222.347   -5547.637986.424 -2058.314
Predicted digit is 5
```
