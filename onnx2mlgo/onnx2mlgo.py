import writer
from pathlib import Path
import onnx
import click

@click.version_option("0.1.1", prog_name="onnx2mlgo")
@click.command()
@click.argument("onnx_path")
def cli(onnx_path):
  # TODO: make sure that model_name is a valid file_name
  # TODO: get rid of everything that isn't required in this repository
  # TODO: remove model_name from the entire transpiler. this is not required and just adds additional complexity. make the name default to 'model' and write the model_name at the beginning as a comment

  onnx_model = onnx.load(Path(onnx_path))

  mlgo_model_path = Path('dist/')
  mlgo_model_path.mkdir(parents=True, exist_ok=True)

  with open(mlgo_model_path / 'test.go', 'w') as file:
    writer.create_go_boilerplate(file)
    writer.create_model_utils(file)
    writer.create_hparams_type(file)
    writer.create_model_type(file)
    writer.create_weight_loading_func(file)
    writer.create_eval_func(file, onnx_model)
    writer.create_main_func(file)
    
  click.echo(click.style(f"Transpilation complete!", fg="green"))
  click.echo(f"MLGO file created in {mlgo_model_path.absolute()}.")

if __name__ == "__main__":
  cli()