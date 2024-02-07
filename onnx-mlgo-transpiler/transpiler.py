import writer

def read_from_onnx():
  pass

def transpile(onnx, model_name):
  with open('test.go', 'w') as file:
    writer.create_go_boilerplate(file)
    writer.create_hparams_type(file, model_name)
    writer.create_model_type(file, model_name)
    writer.create_weight_loading_func(file, model_name)
    writer.create_eval_func(file)
    writer.create_main_func(file)  

def main():
  onnx_output = read_from_onnx()
  transpile(onnx_output, 'test')
  
main()