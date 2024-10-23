import json
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Query
from pydantic_settings import BaseSettings
from sklearn.datasets import make_classification # type: ignore


RANDOM_STATE = 42


class Settings(BaseSettings):
  n_samples: int = 1_000
  n_features: int = 10
  n_informative: int = 6


app = FastAPI()
cfg = Settings()


@app.get("/data")
def data(size: Annotated[int| None, Query(gt=1000)] = None):
  X, y = make_classification(
    n_samples=cfg.n_samples if size is None else size, 
    n_features=cfg.n_features,
    n_informative=cfg.n_informative, 
    random_state=RANDOM_STATE,
    class_sep=np.random.uniform(0.7, 1.),
  )
  return {
    'X': X.tolist(),
    'y': y.tolist(),
  }