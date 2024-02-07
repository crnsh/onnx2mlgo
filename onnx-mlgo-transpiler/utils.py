def create_hparams_type(file):
  file.write(
"""
type mnist_hparams struct{
  n_input int32;
  n_hidden int32;
  n_classes int32;
}"""
  )

def create_model_type(file):
  pass

def create_weight_loading_func(file):
  pass

def create_eval_func(file):
  pass

def create_main_func(file):
  pass