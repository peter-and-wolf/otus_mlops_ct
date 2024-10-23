import logging
import requests
import numpy as np
from numpy.typing import NDArray


def get_data(url: str) -> tuple[NDArray, NDArray]:
  r = requests.get(url)
  r.raise_for_status()
  d = r.json()
  return np.array(d['X']), np.array(d['y'])