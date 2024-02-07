import writer
from pathlib import Path

def read_from_onnx():
  pass

def transpile(onnx, model_name):
  # TODO: make sure that model_name is a valid file_name
  # TODO: get rid of everything that isn't required in this repository

  utils_path = Path('dist/utils/')
  mlgo_model_path = Path('dist/mlgo_model/')

  utils_path.mkdir(parents=True, exist_ok=True)
  mlgo_model_path.mkdir(parents=True, exist_ok=True)

  # Transpilation Begins
  with open(utils_path / 'utils.go', 'w') as file:
    writer.create_model_utils(file)

  with open(mlgo_model_path / 'test.go', 'w') as file:
    writer.create_go_boilerplate(file)
    writer.create_hparams_type(file, model_name)
    writer.create_model_type(file, model_name)
    writer.create_weight_loading_func(file, model_name)
    writer.create_eval_func(file, model_name)
    writer.create_main_func(file, model_name)

def main():
  onnx_output = read_from_onnx()
  transpile(onnx_output, 'test')
  
main()