from typing import Annotated

import typer
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score # type:ignore

from utils import get_data

from cfg import EVALUATION_EXPERIMENT_NAME, MODEL_NAME, PROD_ALIAS


def eval(name: str, alias: str, X: NDArray, y: NDArray) -> float:
  model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
  score = accuracy_score(model.predict(X), y)
  
  client = MlflowClient()
  mlflow.set_experiment(EVALUATION_EXPERIMENT_NAME)
  with mlflow.start_run():  
    version = client.get_model_version_by_alias(name, alias)
    mlflow.log_params({
      'version': version.version,
      'run_id': version.run_id,
    }) 
    mlflow.log_metric('acuracy', score)
  
  return score


def main(alias: Annotated[str, typer.Option()] = PROD_ALIAS,
         url: Annotated[str, typer.Option()] = 'http://localhost:8000/data') -> None:
  try:
    X, y = get_data(url)
    score = eval(MODEL_NAME, alias, X, y)
    print(f'{MODEL_NAME}@{alias} shows {score}')
  except Exception as x:
    print(x)



if __name__ == '__main__':
  typer.run(main)
