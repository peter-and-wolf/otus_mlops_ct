import mlflow
from typing import Annotated, Any
from itertools import product

import typer
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression # type:ignore
from sklearn.tree import DecisionTreeClassifier # type:ignore
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.metrics import accuracy_score # type:ignore

from utils import get_data
from cfg import ARTIFACT_PATH, TRAINING_EXPERIMENT_NAME


# TODO: It's not the best way to do hyperparameters search. 
# Want to know how to do better, sign up for the course.
models = [
  {
    'cls': LogisticRegression,
    'params': {
      'solver': ['liblinear', 'newton-cholesky'],
      'C': [.1, .5, 1., 2.]
    }
  },
  {
    'cls': DecisionTreeClassifier,
    'params': {
      'criterion': ['gini', 'entropy', 'log_loss'],
      'max_depth': [3, 5, 7, 9, 11]
    }
  }
]


def get_params(p: dict[Any, Any]) -> list[dict[Any, Any]]: 
  return [
    dict(map(lambda i,j : (i,j), p.keys(), t)) for t in product(*p.values())
  ]


def train(X: NDArray, y: NDArray) -> None:
  
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)
  
  for m in models:
    model_class = m['cls']
    for params in get_params(m['params']):
      cls = model_class(**params)
      cls.fit(X_train, y_train)
      score = accuracy_score(
        cls.predict(X_test), y_test
      )
      print(f'{cls} shows {score}')

      with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param('cls', cls.__class__.__name__)
        mlflow.log_metric('accuracy', score)
        mlflow.sklearn.log_model(cls, ARTIFACT_PATH)


def main(url: Annotated[str, typer.Option()] = 'http://localhost:8000/data') -> None:
  X, y = get_data(url)
  train(X, y)


if __name__ == '__main__':
  typer.run(main)


