"""Modelos de classificação para prever Sleep_Quality."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    COMPARACAO_MODELOS_PATH,
    GRAFICOS_DIR,
    ID_COL,
    MELHOR_MODELO_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from .preprocessamento import executar_preprocessamento


FEATURE_NUMERICAS = [
    "Age",
    "Coffee_Intake",
    "Caffeine_mg",
    "Sleep_Hours",
    "BMI",
    "Heart_Rate",
    "Physical_Activity_Hours",
    "Smoking",
    "Alcohol_Consumption",
    "Caffeine_per_Cup",
    "Risk_Habits",
]

FEATURE_CATEGORICAS = [
    "Gender",
    "Country",
    "Stress_Level",
    "Health_Issues",
    "Occupation",
    "Age_Group",
]


def preparar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo, removendo identificadores sem valor preditivo."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Coluna alvo '{TARGET_COL}' não encontrada.")

    y = df[TARGET_COL].astype(str)
    # Sleep_Quality já é categórica no dataset; mantemos as classes originais.
    cols = [c for c in FEATURE_NUMERICAS + FEATURE_CATEGORICAS if c in df.columns]
    x = df[cols].copy()

    # Garante que ID nunca entre nas features
    x = x.drop(columns=[ID_COL], errors="ignore")
    return x, y


def construir_preprocessador(x: pd.DataFrame) -> ColumnTransformer:
    """Cria transformações ajustáveis apenas no treino (anti data leakage)."""
    num_cols = [c for c in FEATURE_NUMERICAS if c in x.columns]
    cat_cols = [c for c in FEATURE_CATEGORICAS if c in x.columns]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def avaliar_modelo(nome: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    """Treina, avalia e retorna métricas de treino/teste + CV."""
    pipeline.fit(x_train, y_train)

    pred_train = pipeline.predict(x_train)
    pred_test = pipeline.predict(x_test)

    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="f1_weighted")

    metricas = {
        "modelo": nome,
        "accuracy_treino": accuracy_score(y_train, pred_train),
        "accuracy_teste": accuracy_score(y_test, pred_test),
        "precision_teste": precision_score(y_test, pred_test, average="weighted", zero_division=0),
        "recall_teste": recall_score(y_test, pred_test, average="weighted", zero_division=0),
        "f1_teste": f1_score(y_test, pred_test, average="weighted", zero_division=0),
        "f1_cv_media": float(cv_scores.mean()),
        "f1_cv_std": float(cv_scores.std()),
        "classification_report_teste": classification_report(y_test, pred_test, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred_test),
        "classes": sorted(y_test.unique()),
        "y_pred_teste": pred_test,
        "pipeline": pipeline,
    }

    gap = metricas["accuracy_treino"] - metricas["accuracy_teste"]
    if gap > 0.10:
        metricas["diagnostico"] = "Possível overfitting (treino bem acima do teste)."
    elif metricas["accuracy_teste"] < 0.45 and len(y_test.unique()) > 2:
        metricas["diagnostico"] = "Possível underfitting (desempenho baixo no teste)."
    else:
        metricas["diagnostico"] = "Desempenho coerente entre treino e teste."

    return metricas


def salvar_matriz_confusao(metricas: dict, nome_arquivo: str) -> Path:
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        metricas["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metricas["classes"],
        yticklabels=metricas["classes"],
        ax=ax,
    )
    ax.set_title(f"Matriz de confusão — {metricas['modelo']}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()
    destino = GRAFICOS_DIR / nome_arquivo
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


def treinar_e_comparar(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Treina Random Forest e Logistic Regression, compara e salva o melhor."""
    if df is None:
        df = executar_preprocessamento()

    x, y = preparar_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessador = construir_preprocessador(x)

    modelos = {
        "Random Forest": Pipeline(
            steps=[
                ("prep", preprocessador),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", construir_preprocessador(x)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    resultados = []
    detalhes = {}
    for nome, pipe in modelos.items():
        m = avaliar_modelo(nome, pipe, x_train, x_test, y_train, y_test)
        detalhes[nome] = m
        resultados.append(
            {
                "modelo": nome,
                "accuracy_treino": round(m["accuracy_treino"], 4),
                "accuracy_teste": round(m["accuracy_teste"], 4),
                "precision_teste": round(m["precision_teste"], 4),
                "recall_teste": round(m["recall_teste"], 4),
                "f1_teste": round(m["f1_teste"], 4),
                "f1_cv_media": round(m["f1_cv_media"], 4),
                "f1_cv_std": round(m["f1_cv_std"], 4),
                "diagnostico": m["diagnostico"],
            }
        )
        arquivo_cm = (
            "16_matriz_confusao_random_forest.png"
            if "Forest" in nome
            else "17_matriz_confusao_logistic_regression.png"
        )
        salvar_matriz_confusao(m, arquivo_cm)
        print(f"\n===== {nome} =====")
        print(m["classification_report_teste"])
        print("Diagnóstico:", m["diagnostico"])

    comparacao = pd.DataFrame(resultados).sort_values("f1_teste", ascending=False)
    COMPARACAO_MODELOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparacao.to_csv(COMPARACAO_MODELOS_PATH, index=False)

    melhor_nome = comparacao.iloc[0]["modelo"]
    melhor = detalhes[melhor_nome]
    joblib.dump(melhor["pipeline"], MELHOR_MODELO_PATH)

    meta = {
        "melhor_modelo": melhor_nome,
        "f1_teste": float(comparacao.iloc[0]["f1_teste"]),
        "accuracy_teste": float(comparacao.iloc[0]["accuracy_teste"]),
        "caminho_modelo": str(MELHOR_MODELO_PATH),
    }
    with open(COMPARACAO_MODELOS_PATH.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== Comparação de modelos ===")
    print(comparacao.to_string(index=False))
    print(f"\nMelhor modelo: {melhor_nome}")
    print(f"Salvo em: {MELHOR_MODELO_PATH}")
    return comparacao, detalhes


if __name__ == "__main__":
    treinar_e_comparar()
