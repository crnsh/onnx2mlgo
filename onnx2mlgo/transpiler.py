import codegen
from pathlib import Path
import onnx
import click
from graph import Graph

@click.version_option("0.1.2", prog_name="onnx2mlgo")
@click.command()
@click.argument(
  "onnx_path",
  type=click.Path(
    exists=True,
    file_okay=True,
    readable=True,
    path_type=Path,
  )
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
  # TODO: make sure that model_name is a valid file_name
  # TODO: get rid of everything that isn't required in this repository
  # TODO: remove model_name from the entire transpiler. this is not required and just adds additional complexity. make the name default to 'model' and write the model_name at the beginning as a comment

  onnx_model = onnx.load(Path(onnx_path))
  graph = Graph(onnx_model)

  mlgo_model_path = Path(output_dir)
  mlgo_model_path.mkdir(parents=True, exist_ok=True)

  weight_file = 'model-weights-f32.bin'

  with open(mlgo_model_path / 'test.go', 'w') as file:
    codegen.create_go_boilerplate(file)
    codegen.create_model_utils(file)
    codegen.create_eval_func(file, graph)
    codegen.create_main_func(file)

  click.echo(click.style(f"Transpilation complete!", fg="green"))
  click.echo(f"MLGO file created in {mlgo_model_path.absolute()}.")

if __name__ == "__main__":
  cli()