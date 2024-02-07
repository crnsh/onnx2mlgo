import utils

def read_from_onnx():
  pass

def transpile(onnx):
  with open('test.go', 'w') as file:    
    # type mnist_model
    utils.create_model_type(file)

    # func mnist_model_load
    utils.create_weight_loading_func(file)

    # func mnist_eval
    utils.create_eval_func(file)
    
    # func TestMNIST
    utils.create_main_func(file)
  

def main():
  onnx_output = read_from_onnx()
  transpile(onnx_output)
  
main()