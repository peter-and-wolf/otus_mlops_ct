from typing import Annotated

import typer

from eval import eval
from train import train
from register import register
from utils import get_data
from cfg import MODEL_NAME, PROD_ALIAS


def main(thresh: Annotated[float, typer.Option(min=.0, max=1.)] = .8,
         url: Annotated[str, typer.Option()] = 'http://localhost:8000/data') -> None:
  
  X, y = get_data(url)

  score = eval(MODEL_NAME, PROD_ALIAS, X, y)
  print(f'{MODEL_NAME}@{PROD_ALIAS} shows {score}')

  if score < thresh:
    print("and it's too bad...")
    train(X, y)
    register(thresh)
  else:
    print("and it's ok can get by for a while...")


if __name__ == '__main__':
  typer.run(main)