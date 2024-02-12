def create_graph(onnx):

  # assert : output is a graph where each node a valid mlgo operation.

  return output 

def create_single_layer(var_name: str, mlgo_op: str, input_list):

  # TODO: check whether the number of inputs are valid for the given mlgo_op
  
  input_str = ', '.join(input_list)
  return f'{var_name} := ml.{mlgo_op}(ctx0, {input_str})'

def create_layers(onnx):
  
  mlgo_graph = create_graph(onnx)
  output = ""

  i = 1
  for node in mlgo_graph.in_order:
    
    # TODO: make sure that the order of inputs, weights and biases is correct

    input_list = [INPUT STRING, weights and biases]

    output.append(create_single_layer(f'l{x}', node.mlgo_op, input_list))

  # assert : output is the go language output for the layers
  
  output = """\
  // fc1 MLP = Ax + b
  fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
  fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)

  // softmax
  final := ml.SoftMax(ctx0, fc2)  
"""
  
  return output