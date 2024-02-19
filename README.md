<div align="center">
  <img src="https://github.com/crnsh/onnx2mlgo/assets/79533543/a7c3c0e1-277d-4079-b827-7ae2fb566493" width=500>
</div>
<hr>

onnx2mlgo is an ONNX to MLGO transpiler.

Features:
* Single CLI command to output Go files
* Compatibility Checker shows all missing operators
* Go code generator

Currently this transpiler only transpiles a limited number of models. The CLI tells you which operations are missing and need to be implemented. Also, only ONNX models with initialized weights are supported.

You will have to manually import the input data in the generated Go file.

If you find a bug, please [let me know](https://github.com/crnsh/onnx2mlgo/issues)!

## Installation
```bash
git pull git@github.com:crnsh/onnx2mlgo.git
cd onnx2mlgo
python3 -m pip install -r requirements.txt
```

## Usage
Make sure you have Go installed!

```bash
python3 onnx2mlgo/transpiler.py tests/onnx_fc.onnx -o mlgo/dist
cd mlgo/dist
go run test.go
```
