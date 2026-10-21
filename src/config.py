"""Caminhos e constantes compartilhadas do projeto Health&Life Analytics."""

from pathlib import Path

# Raiz do projeto (pasta healthlife_analytics/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "synthetic_coffee_health_10000.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "coffee_health_processed.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
GRAFICOS_DIR = RESULTS_DIR / "graficos"
ESTATISTICAS_PATH = RESULTS_DIR / "estatisticas.csv"
COMPARACAO_MODELOS_PATH = RESULTS_DIR / "comparacao_modelos.csv"
MELHOR_MODELO_PATH = RESULTS_DIR / "melhor_modelo.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colunas numéricas e categóricas esperadas no dataset bruto
NUMERIC_COLS = [
    "Age",
    "Coffee_Intake",
    "Caffeine_mg",
    "Sleep_Hours",
    "BMI",
    "Heart_Rate",
    "Physical_Activity_Hours",
    "Smoking",
    "Alcohol_Consumption",
]

CATEGORICAL_COLS = [
    "Gender",
    "Country",
    "Sleep_Quality",
    "Stress_Level",
    "Health_Issues",
    "Occupation",
]

TARGET_COL = "Sleep_Quality"
ID_COL = "ID"
