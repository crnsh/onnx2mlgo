import writer
from pathlib import Path
import onnx

def read_from_onnx():
  onnx_path = Path('./mnist_test.onnx')
  model = onnx.load(onnx_path)

  return model

def transpile(onnx_model, model_name):
  # TODO: make sure that model_name is a valid file_name
  # TODO: get rid of everything that isn't required in this repository

  mlgo_model_path = Path('dist/')

  mlgo_model_path.mkdir(parents=True, exist_ok=True)

  with open(mlgo_model_path / 'test.go', 'w') as file:
    writer.create_go_boilerplate(file)
    writer.create_model_utils(file)
    writer.create_hparams_type(file, model_name)
    writer.create_model_type(file, model_name)
    writer.create_weight_loading_func(file, model_name)
    writer.create_eval_func(file, model_name)
    writer.create_main_func(file, model_name)

def main():
  onnx_model = read_from_onnx()
  transpile(onnx_model, 'test')
  
main()