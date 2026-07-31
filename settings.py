import os
import pathlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FREIBURG_ID = 34
BASE_DIR = str(pathlib.Path().resolve())
IMPECT_DIR = os.path.join(BASE_DIR, 'open-data', 'data')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PCT_BY_LINE = True
RANDOM_STATE = 308
PLOTLY_CLUSTER = True