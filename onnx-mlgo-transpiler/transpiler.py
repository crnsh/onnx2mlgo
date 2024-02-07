import writer

def read_from_onnx():
  pass

def transpile(onnx, model_name):
  with open('test.go', 'w') as file:
    # imports and boilerplate
    writer.create_go_boilerplate(file)

    # type mnist_hparams
    writer.create_hparams_type(file, model_name)
    
    # type mnist_model
    writer.create_model_type(file, model_name)

    # func mnist_model_load
    writer.create_weight_loading_func(file)

    # func mnist_eval
    writer.create_eval_func(file)

    # func TestMNIST
    writer.create_main_func(file)
  

def main():
  onnx_output = read_from_onnx()
  transpile(onnx_output, 'mnist')
  
main()