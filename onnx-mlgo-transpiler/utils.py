def create_layers():
  
  # assert : output is the go language output for the layers
  
  output = """\
  // fc1 MLP = Ax + b
  fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
  fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)

  // softmax
  final := ml.SoftMax(ctx0, fc2)  
"""
  
  return output