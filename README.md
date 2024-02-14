# CLI Transpiler from ONNX to MLGO

Compiles ONNX to MLGO. MLGO is tensor library for machine learning in pure Golang that can run on MIPS. 

The machine learning part of this project refers to the legendary [ggml.cpp](https://github.com/ggerganov/ggml) framework.

Located in `onnx2mlgo`.

1. `cd onnx2mlgo`
2. `python3 transpiler.py`

## Build

`pip install -r requirements.txt`