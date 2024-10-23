from typing import Annotated

import mlflow
from mlflow import MlflowClient
import typer

from cfg import ARTIFACT_PATH, MODEL_NAME, CHAMP_ALIAS, TRAINING_EXPERIMENT_NAME


def register(thresh: float) -> None:
  runs = mlflow.search_runs(
    experiment_names=[TRAINING_EXPERIMENT_NAME], 
    filter_string=f'metrics.accuracy > {thresh}',
    order_by=['metrics.accuracy DESC'],
    output_format='list'
  )
  if len(runs) > 0:
    client = MlflowClient()

    best = runs[0]
    model_uri = f'runs:/{best.info.run_id}/{ARTIFACT_PATH}'  
    mv = client.create_model_version(MODEL_NAME, model_uri, best.info.run_id)
    client.set_registered_model_alias(MODEL_NAME, CHAMP_ALIAS, mv.version)
    print(f'{MODEL_NAME} v{mv.version} with {best.data.metrics['accuracy']} has been registred')
  else:
    print('ALARM ALARM ALARM!')  


def main(thresh: Annotated[float, typer.Option(min=.0, max=1.)] = .8) -> None:
  register(thresh)


if __name__ == '__main__':
  typer.run(main)