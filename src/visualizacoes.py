"""Geração de visualizações exploratórias e comparativas do projeto."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import GRAFICOS_DIR
from .preprocessamento import executar_preprocessamento

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def _salvar(fig: plt.Figure, nome: str) -> Path:
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    destino = GRAFICOS_DIR / nome
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


def histograma_distribuicoes(df: pd.DataFrame) -> Path:
    cols = [
        "Age",
        "Coffee_Intake",
        "Caffeine_mg",
        "Sleep_Hours",
        "BMI",
        "Heart_Rate",
        "Physical_Activity_Hours",
    ]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2a6f97")
        axes[i].set_title(f"Distribuição de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequência")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distribuições das variáveis numéricas principais", fontsize=14)
    fig.tight_layout()
    return _salvar(fig, "01_histogramas_numericas.png")


def boxplots_principais(df: pd.DataFrame) -> Path:
    cols = ["Coffee_Intake", "Caffeine_mg", "Sleep_Hours", "BMI", "Heart_Rate"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#61a5c2")
        ax.set_title(col)
        ax.set_ylabel(col)
    fig.suptitle("Boxplots — identificação visual de outliers", fontsize=13)
    fig.tight_layout()
    return _salvar(fig, "02_boxplots_principais.png")


def barras_e_pizza_categoricas(df: pd.DataFrame) -> list[Path]:
    caminhos = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ordem = df["Sleep_Quality"].value_counts().index
    sns.countplot(
        data=df,
        x="Sleep_Quality",
        order=ordem,
        hue="Sleep_Quality",
        legend=False,
        ax=ax,
        palette="Blues",
    )
    ax.set_title("Contagem de Sleep_Quality")
    ax.set_xlabel("Qualidade do sono")
    ax.set_ylabel("Quantidade")
    caminhos.append(_salvar(fig, "03_barras_sleep_quality.png"))

    fig, ax = plt.subplots(figsize=(6, 6))
    contagem = df["Stress_Level"].value_counts()
    ax.pie(contagem, labels=contagem.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Proporção de Stress_Level")
    caminhos.append(_salvar(fig, "04_pizza_stress_level.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="Gender", hue="Gender", legend=False, ax=ax, palette="pastel")
    ax.set_title("Distribuição por gênero")
    ax.set_xlabel("Gênero")
    ax.set_ylabel("Quantidade")
    caminhos.append(_salvar(fig, "05_barras_genero.png"))
    return caminhos


def matriz_correlacao(df: pd.DataFrame) -> Path:
    num = df.select_dtypes(include="number").drop(columns=["ID"], errors="ignore")
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlação (variáveis numéricas)")
    fig.tight_layout()
    return _salvar(fig, "06_matriz_correlacao.png")


def cafe_vs_sono(df: pd.DataFrame) -> list[Path]:
    caminhos = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ordem = ["Poor", "Fair", "Good", "Excellent"]
    ordem = [o for o in ordem if o in df["Sleep_Quality"].unique()] + [
        x for x in df["Sleep_Quality"].unique() if x not in ["Poor", "Fair", "Good", "Excellent"]
    ]
    sns.boxplot(data=df, x="Sleep_Quality", y="Coffee_Intake", order=ordem, ax=ax)
    ax.set_title("Consumo de café versus qualidade do sono")
    ax.set_xlabel("Qualidade do sono")
    ax.set_ylabel("Coffee_Intake")
    caminhos.append(_salvar(fig, "07_cafe_vs_sleep_quality.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="Caffeine_mg",
        y="Sleep_Hours",
        hue="Stress_Level",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Cafeína versus horas de sono (amostra)")
    ax.set_xlabel("Caffeine_mg")
    ax.set_ylabel("Sleep_Hours")
    ax.legend(title="Stress_Level")
    caminhos.append(_salvar(fig, "08_cafeina_vs_sleep_hours.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Stress_Level", y="Caffeine_mg", ax=ax)
    ax.set_title("Cafeína versus nível de estresse")
    ax.set_xlabel("Stress_Level")
    ax.set_ylabel("Caffeine_mg")
    caminhos.append(_salvar(fig, "09_cafeina_vs_stress.png"))
    return caminhos


def sono_estresse_atividade(df: pd.DataFrame) -> list[Path]:
    caminhos = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Sleep_Quality", y="Sleep_Hours", ax=ax)
    ax.set_title("Horas de sono versus qualidade do sono")
    ax.set_xlabel("Sleep_Quality")
    ax.set_ylabel("Sleep_Hours")
    caminhos.append(_salvar(fig, "10_sleep_hours_vs_quality.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    tabela = pd.crosstab(df["Stress_Level"], df["Sleep_Quality"], normalize="index")
    tabela.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Estresse versus qualidade do sono (proporção)")
    ax.set_xlabel("Stress_Level")
    ax.set_ylabel("Proporção")
    ax.legend(title="Sleep_Quality", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    caminhos.append(_salvar(fig, "11_stress_vs_sleep_quality.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Sleep_Quality", y="Physical_Activity_Hours", ax=ax)
    ax.set_title("Atividade física versus qualidade do sono")
    ax.set_xlabel("Sleep_Quality")
    ax.set_ylabel("Physical_Activity_Hours")
    caminhos.append(_salvar(fig, "12_atividade_vs_sleep_quality.png"))
    return caminhos


def demografia_e_saude(df: pd.DataFrame) -> list[Path]:
    caminhos = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ordem_idade = ["18-24", "25-34", "35-44", "45-54", "55+"]
    ordem_idade = [o for o in ordem_idade if o in df["Age_Group"].unique()]
    sns.barplot(
        data=df,
        x="Age_Group",
        y="Coffee_Intake",
        order=ordem_idade,
        estimator=np.mean,
        errorbar="sd",
        ax=ax,
        color="#014f86",
    )
    ax.set_title("Consumo médio de café por faixa etária")
    ax.set_xlabel("Faixa etária")
    ax.set_ylabel("Coffee_Intake (média ± DP)")
    caminhos.append(_salvar(fig, "13_idade_vs_cafe.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Gender", y="Coffee_Intake", ax=ax)
    ax.set_title("Consumo de café por gênero")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Coffee_Intake")
    caminhos.append(_salvar(fig, "14_genero_vs_cafe.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="Health_Issues", y="Coffee_Intake", ax=axes[0])
    axes[0].set_title("Café versus problemas de saúde")
    axes[0].set_xlabel("Health_Issues")
    axes[0].set_ylabel("Coffee_Intake")

    sns.boxplot(data=df, x="Health_Issues", y="Heart_Rate", ax=axes[1])
    axes[1].set_title("Frequência cardíaca versus problemas de saúde")
    axes[1].set_xlabel("Health_Issues")
    axes[1].set_ylabel("Heart_Rate")
    fig.tight_layout()
    caminhos.append(_salvar(fig, "15_saude_vs_indicadores.png"))
    return caminhos


def gerar_todas_visualizacoes(df: pd.DataFrame | None = None) -> list[Path]:
    """Gera e salva todos os gráficos obrigatórios do projeto."""
    if df is None:
        df = executar_preprocessamento()

    caminhos: list[Path] = []
    caminhos.append(histograma_distribuicoes(df))
    caminhos.append(boxplots_principais(df))
    caminhos.extend(barras_e_pizza_categoricas(df))
    caminhos.append(matriz_correlacao(df))
    caminhos.extend(cafe_vs_sono(df))
    caminhos.extend(sono_estresse_atividade(df))
    caminhos.extend(demografia_e_saude(df))
    print(f"{len(caminhos)} gráficos salvos em: {GRAFICOS_DIR}")
    return caminhos


if __name__ == "__main__":
    gerar_todas_visualizacoes()
