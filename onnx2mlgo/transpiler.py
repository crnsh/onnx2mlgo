import codegen
from pathlib import Path
import onnx, onnx.numpy_helper
import utils
import click
from graph import Graph
import struct


@click.version_option("0.1.2", prog_name="onnx2mlgo")
@click.command()
@click.argument(
    "onnx_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(
        exists=False,
        path_type=Path,
    ),
    default="dist",
)
def cli(onnx_path, output_dir):
    onnx_model = onnx.load(Path(onnx_path))
    onnx_model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)

    graph = Graph(onnx_model_with_shapes)

    mlgo_model_path = Path(output_dir)
    mlgo_model_path.mkdir(parents=True, exist_ok=True)

    weight_file_folder = mlgo_model_path / Path("models")
    weight_file_folder.mkdir(parents=True, exist_ok=True)

    weight_file = weight_file_folder / Path("model-weights-f32.bin")

    # create model weight file
    with open(weight_file, "wb") as file:
        file.write(struct.pack("i", 0x6D6C676F))
        for initializer in onnx_model_with_shapes.graph.initializer:
            weight = onnx.numpy_helper.to_array(initializer)
            weight.astype(">f4")
            weight.tofile(file)

    # create go file
    with open(mlgo_model_path / "model.go", "w") as file:
        # TODO: currently supports single input tensor (input tensors aren't weights). extend this later
        input_data_shape = graph.inputs[0].get_shape()
        input_data_var = "inputData"
        input_dtype = "float32"
        codegen.create_go_boilerplate_and_model_utils(file)
        codegen.create_eval_func(file, graph, input_data_var, input_dtype)
        codegen.create_main_func(
            file, weight_file.absolute(), input_data_var, input_data_shape, input_dtype
        )

    click.echo(click.style(f"Transpilation complete!", fg="green"))
    click.echo(f"MLGO file created in {mlgo_model_path.absolute()}.")


if __name__ == "__main__":
    cli()
