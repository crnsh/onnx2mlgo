import utils

def read_from_onnx():
  pass

def transpile(onnx):
  
  # type mnist_model
  utils.create_model_type()
  
  # func mnist_model_load
  utils.create_weight_loading_func()
  
  # func mnist_eval
  utils.create_eval_func()
  
  # func TestMNIST
  utils.create_main_func()
  

def main():
  onnx_output = read_from_onnx()
  
  transpile(onnx_output)
  
main()