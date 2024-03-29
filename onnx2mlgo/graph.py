from onnx.checker import check_model
from onnx import numpy_helper
from typing import List, Set
import utils

EXIT_ON_UNSUPPORTED_OP = False


class Node:
    unsupported_ops: Set[str] = set()
    temp_cnt = 0

    def __init__(self, mlgo_op: str, inputs: List[str], output: str):
        self._op = mlgo_op
        self._inputs = inputs
        self._output = output
        self._input_first = True

    @property
    def input_first(self):
        return self._input_first

    @property
    def op(self):
        return self._op

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    @classmethod
    def create_node(cls, onnx_node, onnx_model) -> List:
        node_inputs = utils.sanitize_list(onnx_node.input)
        node_output = utils.sanitize_string(onnx_node.output[0])
        op = onnx_node.op_type
        if len(onnx_node.output) > 1:
            raise NotImplementedError(
                f"onnx nodes with multiple outputs are currently not supported"
            )
        if op == "Gemm":
            temp_output = f"temp{Node.temp_cnt}"
            Node.temp_cnt += 1
            node1 = Node("MulMat", [node_inputs[1], node_inputs[0]], temp_output)
            node2 = Node("Add", [temp_output, node_inputs[2]], node_output)
            return [node1, node2]

        # Binary Ops

        elif op == "Add":
            node = Node("Add", node_inputs, node_output)
            return [node]
        elif op == "Sub":
            node = Node("Sub", node_inputs, node_output)
            return [node]
        elif op == "Mul":
            node = Node("Mul", node_inputs, node_output)
            return [node]
        elif op == "Div":
            node = Node("Div", node_inputs, node_output)
            return [node]
        elif op == "Pow":
            node = Node("Pow", node_inputs, node_output)
            return [node]
        elif op == "MatMul":
            node = Node("MulMat", node_inputs, node_output)
            return [node]

        # Unary Ops
        elif op == "Constant":
            # for some reason, shape inference does not consistently work on constants, so all this code is just trying to figure out the shape of the other inputs
            # TODO: write a better comment for this
            op_inputs = None
            for node in onnx_model.graph.node:
                if onnx_node.output[0] in node.input:
                    # print(onnx_node.output[0], node.input)
                    op_inputs = node.input

            inputs_excluding_constant = list(
                filter(lambda x: x != onnx_node.output[0], op_inputs)
            )
            if len(inputs_excluding_constant) > 1:
                raise NotImplementedError(
                    "Constants are only supported with binary ops."
                )
            other_input = inputs_excluding_constant[0]
            for shape_info in onnx_model.graph.value_info:
                if shape_info.name == other_input:
                    shape = utils.get_shape_from_shape_proto(
                        shape_info.type.tensor_type.shape
                    )

            rank = len(shape)
            tensor_variant = utils.tensor_variants[rank]
            tensor_def = utils.define_tensor(
                node_output, tensor_variant, "nil", "TYPE_F32", shape
            )
            if len(onnx_node.attribute) > 1:
                raise NotImplementedError("More than a single attribute not supported")
            const_val = numpy_helper.to_array(onnx_node.attribute[0].t)
            tensor_init = utils.initialize_const_tensor_for_loop(
                "i", node_output, const_val
            )
            return [tensor_def, tensor_init]
        elif op == "Sqrt":
            node = Node("Sqrt", node_inputs, node_output)
            return [node]
        elif op == "Relu":
            node = Node("Relu", node_inputs, node_output)
            return [node]
        elif op == "Softmax":
            node = Node("SoftMax", node_inputs, node_output)
            return [node]
        elif op == "Erf":
            node = Node("Erf", node_inputs, node_output)
            return [node]
        else:
            Node.unsupported_ops.add(op) if op not in Node.unsupported_ops else None
            if EXIT_ON_UNSUPPORTED_OP:
                raise NotImplementedError(
                    f"{onnx_node.op_type} has not been implemented yet."
                )
            return []


class Input:
    def __init__(self, onnx_input):
        self._name = utils.sanitize_string(onnx_input.name)
        self._type = onnx_input.type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def get_shape(self):
        shape = utils.get_shape_from_shape_proto(self.type.tensor_type.shape)
        return shape


class Initializer:
    def __init__(self, onnx_initializer):
        self._dims = onnx_initializer.dims
        self._data_type = onnx_initializer.data_type
        self._name = utils.sanitize_string(onnx_initializer.name)
        self._raw_data = onnx_initializer.raw_data

    @property
    def dims(self):
        return self._dims

    @property
    def data_type(self):
        return self._data_type

    @property
    def name(self):
        return self._name

    @property
    def raw_data(self):
        return self._raw_data


class Graph:
    def __init__(self, onnx_model):
        # TODO: sanitize all strings in input, output and initializer. might need to create separate classes for input, initializers and output
        check_model(onnx_model)
        self._nodes: List[Node] = []
        self._inputs: List[Input] = []  # type List[Tensor]
        self._initializers: List[Initializer] = []  # type List[Tensor]
        self._outputs = onnx_model.graph.output  # type List[Tensor]
        for onnx_node in onnx_model.graph.node:
            # assert : self._graph is a complete graph with last output as last_output
            # assert : node takes in 1 input, x weights and has 1 output
            node_list = Node.create_node(onnx_node, onnx_model)
            for node in node_list:
                self.add_node(node)
        for onnx_input in onnx_model.graph.input:
            self._inputs.append(Input(onnx_input))
        for onnx_initializer in onnx_model.graph.initializer:
            self._initializers.append(Initializer(onnx_initializer))
        if len(Node.unsupported_ops) > 0:
            raise NotImplementedError(
                f"Some operations {Node.unsupported_ops} are not currently supported."
            )

    @property
    def nodes(self):
        return self._nodes

    @property
    def inputs(self):
        return self._inputs

    @property
    def initializers(self):
        return self._initializers

    @property
    def outputs(self):
        return self._outputs

    def add_node(self, node):
        self._nodes.append(node)
