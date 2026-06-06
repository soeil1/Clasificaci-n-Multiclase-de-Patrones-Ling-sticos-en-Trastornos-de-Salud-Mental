"""Parametros del proyecto en un solo lugar."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.environ.get("APIT_DATA", os.path.join(ROOT, "data", "raw"))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "reports", "figuras")

RANDOM_SEED = 42
LABELS = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]

# VSM
SUBLINEAR_TF = True
MAX_FEATURES = 5000      # tamano maximo del vocabulario
MIN_DF = 3               # la palabra debe salir en al menos 3 documentos

# Clasificador
TEST_FRACTION = 0.30
NEUTRAL_LABEL = "Neutral"
DEFAULT_THRESHOLD = 0.08

# Nombres de los CSV de entrada
CSV_DEPRESION = "Depression.csv"
CSV_SUICIDA = "Suicadal_tendencies_data.csv"
CSV_NEUTRAL = "Neutral.csv"
CSV_ESQUIZOFRENIA = "Reddit-Based_Schizophrenia_Detection_Dataset.csv"
