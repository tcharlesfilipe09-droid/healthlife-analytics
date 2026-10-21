"""Gera o notebook Jupyter acadêmico completo do projeto."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "analise_completa.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """# Health&Life Analytics — Análise Completa de Café, Cafeína e Saúde

**Empresa fictícia:** Health&Life Analytics  
**Objetivo:** analisar relações entre consumo de café, cafeína, sono, estresse e indicadores de saúde, e construir modelos preditivos para `Sleep_Quality`.

> Observação metodológica: a base é **sintética**. Correlações e desempenho preditivo **não** devem ser interpretados como prova de causalidade clínica.
"""
        ),
        md(
            """## 1. Configuração do ambiente

Carrega bibliotecas e define caminhos relativos à raiz do projeto.
"""
        ),
        code(
            """from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 110

# Garante importação dos módulos locais (src/)
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, DATA_PROCESSED, RESULTS_DIR, GRAFICOS_DIR, RANDOM_STATE
from src.preprocessamento import carregar_dados_brutos, tratar_dados, exportar_processado
from src.analise_exploratoria import (
    resumo_estrutura,
    detectar_outliers_iqr,
    estatisticas_descritivas,
    exportar_estatisticas,
)
from src.visualizacoes import gerar_todas_visualizacoes
from src.machine_learning import treinar_e_comparar, preparar_xy

print("Raiz do projeto:", PROJECT_ROOT)
print("Dataset bruto:", DATA_RAW)
print("Existe?", DATA_RAW.exists())
"""
        ),
        md(
            """## 2. Etapa 1 — Análise Exploratória de Dados (EDA)

### 2.1 Carregamento e inspeção inicial
"""
        ),
        code(
            """df_raw = carregar_dados_brutos()

print("Shape:", df_raw.shape)
print("\\nCinco primeiros registros:")
display(df_raw.head())
print("Cinco últimos registros:")
display(df_raw.tail())
"""
        ),
        code(
            """print("Colunas e tipos:")
print(df_raw.dtypes)
print("\\nValores ausentes:")
print(df_raw.isna().sum())
print("\\nDuplicados:", df_raw.duplicated().sum())
"""
        ),
        md(
            """### 2.2 Dicionário resumido das variáveis

| Coluna | Tipo esperado | Significado |
|---|---|---|
| ID | identificador | Chave única do registro |
| Age | numérico | Idade (anos) |
| Gender | categórico | Gênero |
| Country | categórico | País |
| Coffee_Intake | numérico | Consumo de café (xícaras/unidades) |
| Caffeine_mg | numérico | Cafeína estimada (mg) |
| Sleep_Hours | numérico | Horas de sono |
| Sleep_Quality | categórico | Qualidade do sono (alvo) |
| BMI | numérico | Índice de Massa Corporal |
| Heart_Rate | numérico | Frequência cardíaca |
| Stress_Level | categórico | Nível de estresse |
| Physical_Activity_Hours | numérico | Horas de atividade física |
| Health_Issues | categórico | Problemas de saúde reportados |
| Occupation | categórico | Ocupação |
| Smoking | numérico/binário | Indicador de tabagismo |
| Alcohol_Consumption | numérico/binário | Indicador de consumo de álcool |
"""
        ),
        code(
            """print("Estatísticas descritivas (numéricas):")
display(df_raw.describe().T)

print("\\nDistribuições categóricas principais:")
for col in ["Gender", "Sleep_Quality", "Stress_Level", "Health_Issues", "Occupation"]:
    print(f"\\n--- {col} ---")
    print(df_raw[col].value_counts())
"""
        ),
        md(
            """### 2.3 Limpeza e tratamento

Alterações aplicadas:
1. Remoção de duplicatas (se existirem), para evitar enviesar frequências e modelos.
2. Imputação de ausentes numéricos pela mediana e categóricos pela moda.
3. Criação de variáveis derivadas **sem uso do alvo**:
   - `Caffeine_per_Cup`: intensidade de cafeína por unidade de café.
   - `Age_Group`: faixa etária para análises comparativas.
   - `Risk_Habits`: combinação de tabaco e álcool.
"""
        ),
        code(
            """df = tratar_dados(df_raw)
exportar_processado(df)

estrutura = resumo_estrutura(df)
print(estrutura)
print("\\nOutliers (método IQR):")
display(detectar_outliers_iqr(df))

stats = exportar_estatisticas(df)
display(stats.head(20))
print("Dataset processado salvo em:", DATA_PROCESSED)
"""
        ),
        md(
            """### 2.4 Distribuições e correlações

Os gráficos abaixo investigam idade, café, cafeína, sono, IMC e frequência cardíaca.
"""
        ),
        code(
            """cols = ["Age", "Coffee_Intake", "Caffeine_mg", "Sleep_Hours", "BMI", "Heart_Rate"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for ax, col in zip(axes, cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2a6f97")
    ax.set_title(f"Distribuição de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequência")
fig.suptitle("Histogramas das variáveis-chave")
fig.tight_layout()
plt.show()
print("Interpretação: as distribuições ajudam a identificar assimetria, concentração central e possíveis caudas longas (outliers).")
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 5, figsize=(14, 4))
for ax, col in zip(axes, ["Coffee_Intake", "Caffeine_mg", "Sleep_Hours", "BMI", "Heart_Rate"]):
    sns.boxplot(y=df[col], ax=ax, color="#61a5c2")
    ax.set_title(col)
fig.suptitle("Boxplots — inspeção de outliers")
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="Sleep_Quality", order=df["Sleep_Quality"].value_counts().index, ax=axes[0], palette="Blues")
axes[0].set_title("Barras — Sleep_Quality")
cont = df["Stress_Level"].value_counts()
axes[1].pie(cont, labels=cont.index, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Pizza — Stress_Level")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """num = df.select_dtypes(include="number").drop(columns=["ID"], errors="ignore")
corr = num.corr(numeric_only=True)
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlação (numéricas)")
plt.tight_layout()
plt.show()
print("Interpretação: valores próximos de ±1 indicam associação linear forte; valores próximos de 0 sugerem pouca associação linear.")
print("Lembrete: correlação ≠ causalidade.")
"""
        ),
        md(
            """## 3. Etapa 2 — Visualizações comparativas e insights
"""
        ),
        code(
            """fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.boxplot(data=df, x="Sleep_Quality", y="Coffee_Intake", ax=axes[0, 0])
axes[0, 0].set_title("Café vs qualidade do sono")

sample = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE)
sns.scatterplot(data=sample, x="Caffeine_mg", y="Sleep_Hours", hue="Stress_Level", alpha=0.45, ax=axes[0, 1])
axes[0, 1].set_title("Cafeína vs horas de sono")

sns.boxplot(data=df, x="Stress_Level", y="Caffeine_mg", ax=axes[1, 0])
axes[1, 0].set_title("Cafeína vs estresse")

sns.boxplot(data=df, x="Sleep_Quality", y="Sleep_Hours", ax=axes[1, 1])
axes[1, 1].set_title("Horas de sono vs qualidade")

plt.tight_layout()
plt.show()
"""
        ),
        code(
            """fig, axes = plt.subplots(2, 2, figsize=(12, 9))

tabela = pd.crosstab(df["Stress_Level"], df["Sleep_Quality"], normalize="index")
tabela.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="viridis")
axes[0, 0].set_title("Estresse vs qualidade do sono")
axes[0, 0].legend(title="Sleep_Quality", bbox_to_anchor=(1.02, 1), fontsize=8)

sns.boxplot(data=df, x="Sleep_Quality", y="Physical_Activity_Hours", ax=axes[0, 1])
axes[0, 1].set_title("Atividade física vs qualidade do sono")

ordem_idade = [o for o in ["18-24", "25-34", "35-44", "45-54", "55+"] if o in df["Age_Group"].unique()]
sns.barplot(data=df, x="Age_Group", y="Coffee_Intake", order=ordem_idade, estimator=np.mean, errorbar="sd", ax=axes[1, 0], color="#014f86")
axes[1, 0].set_title("Café médio por faixa etária")

sns.boxplot(data=df, x="Gender", y="Coffee_Intake", ax=axes[1, 1])
axes[1, 1].set_title("Café por gênero")

plt.tight_layout()
plt.show()
"""
        ),
        code(
            """# Exporta todos os gráficos oficiais do projeto para results/graficos/
caminhos = gerar_todas_visualizacoes(df)
print(f"{len(caminhos)} arquivos PNG gerados em {GRAFICOS_DIR}")
"""
        ),
        md(
            """### Principais Descobertas

As descobertas abaixo são geradas a partir dos dados reais carregados nesta execução. Elas descrevem **padrões observados**, não relações causais.
"""
        ),
        code(
            """# Síntese quantitativa para embasar as descobertas
corr_cafe_sono = df[["Coffee_Intake", "Sleep_Hours"]].corr().iloc[0, 1]
corr_caf_sono = df[["Caffeine_mg", "Sleep_Hours"]].corr().iloc[0, 1]
media_cafe_por_sq = df.groupby("Sleep_Quality")["Coffee_Intake"].mean().sort_values(ascending=False)
media_sono_por_stress = df.groupby("Stress_Level")["Sleep_Hours"].mean().sort_values()
media_cafe_genero = df.groupby("Gender")["Coffee_Intake"].mean()
media_cafe_idade = df.groupby("Age_Group")["Coffee_Intake"].mean().reindex(ordem_idade)

print("Correlação Coffee_Intake x Sleep_Hours:", round(corr_cafe_sono, 3))
print("Correlação Caffeine_mg x Sleep_Hours:", round(corr_caf_sono, 3))
print("\\nMédia de café por Sleep_Quality:")
print(media_cafe_por_sq.round(3))
print("\\nMédia de Sleep_Hours por Stress_Level:")
print(media_sono_por_stress.round(3))
print("\\nMédia de café por Gender:")
print(media_cafe_genero.round(3))
print("\\nMédia de café por Age_Group:")
print(media_cafe_idade.round(3))

descobertas = (
    "\\n### Principais Descobertas (texto acadêmico)\\n"
    "1. A associação linear entre cafeína/café e horas de sono foi medida diretamente nos dados (valores impressos acima).\\n"
    "2. Diferenças de consumo médio de café entre classes de Sleep_Quality e faixas etárias podem indicar segmentos úteis para monitoramento populacional.\\n"
    "3. A relação entre estresse e horas/qualidade de sono é relevante para dashboards de bem-estar corporativo da Health&Life Analytics.\\n"
    "4. Indicadores como BMI, Heart_Rate e Health_Issues devem ser analisados em conjunto com hábitos (fumo/álcool), sem inferir causalidade clínica.\\n"
    "5. Como a base é sintética, os padrões servem para demonstração metodológica e prototipagem de produtos analíticos."
)
print(descobertas)
"""
        ),
        md(
            """## 4. Etapa 3 — Machine Learning (classificação de `Sleep_Quality`)

`Sleep_Quality` é tratada como variável **categórica** (classes nominais/ordinais discretas). Não há necessidade de discretização adicional.

Cuidados anti-vazamento:
- `train_test_split` com `random_state=42` e estratificação.
- `Pipeline` + `ColumnTransformer` com scaling/one-hot ajustados apenas no treino.
- Remoção de `ID`.
"""
        ),
        code(
            """x, y = preparar_xy(df)
print("Classes alvo:")
print(y.value_counts())
print("\\nFeatures:", list(x.columns))
print("Shape X:", x.shape)
"""
        ),
        code(
            """comparacao, detalhes = treinar_e_comparar(df)
display(comparacao)

melhor = comparacao.iloc[0]["modelo"]
print("\\nMelhor modelo:", melhor)
print(detalhes[melhor]["classification_report_teste"])
print("Diagnóstico:", detalhes[melhor]["diagnostico"])
print("Modelo salvo em:", RESULTS_DIR / "melhor_modelo.joblib")
"""
        ),
        md(
            """## 5. Etapa 4 — Exportação dos resultados

Arquivos esperados após a execução:
- `data/processed/coffee_health_processed.csv`
- `results/estatisticas.csv`
- `results/comparacao_modelos.csv`
- `results/melhor_modelo.joblib`
- PNGs em `results/graficos/`
"""
        ),
        code(
            """arquivos = [
    DATA_PROCESSED,
    RESULTS_DIR / "estatisticas.csv",
    RESULTS_DIR / "comparacao_modelos.csv",
    RESULTS_DIR / "melhor_modelo.joblib",
]
for arq in arquivos:
    print(("OK" if arq.exists() else "FALTANDO"), "-", arq)

pngs = sorted(GRAFICOS_DIR.glob("*.png"))
print(f"\\nGráficos PNG: {len(pngs)}")
for p in pngs:
    print("-", p.name)
"""
        ),
        md(
            """## 6. Etapa 5 — Conclusões e recomendações

### Objetivos e metodologia
O projeto investigou padrões entre hábitos de café/cafeína, sono, estresse e indicadores de saúde com EDA, visualizações e classificação supervisionada de `Sleep_Quality`.

### Principais achados
As métricas e gráficos gerados nesta execução mostram associações descritivas (não causais) entre hábitos e qualidade do sono. O consumo de café/cafeína, o nível de estresse e a atividade física emergem como dimensões úteis para segmentação analítica.

### Resultados dos algoritmos
A tabela `comparacao_modelos.csv` resume acurácia, precisão, recall e F1 em treino/teste, além de validação cruzada. O melhor modelo foi persistido com Joblib para reutilização.

### Limitações
- Dataset sintético: generalização clínica é limitada.
- Possíveis variáveis omitidas (horário do café, transtornos do sono, medicações).
- Correlação e importância preditiva não equivalem a causalidade.

### Recomendações para a Health&Life Analytics
1. Usar os segmentos de sono/estresse para painéis preventivos, com linguagem cuidadosa (associação, não diagnóstico).
2. Priorizar coleta de variáveis temporais (hora do consumo de cafeína) em bases reais futuras.
3. Monitorar overfitting e reavaliar modelos periodicamente com dados novos.
4. Combinar o classificador de `Sleep_Quality` com regras de negócio e validação de especialistas de saúde.

### Melhorias futuras
- Calibração de probabilidades e análise de importância de variáveis (SHAP).
- Testes de modelos boosting (XGBoost/LightGBM) e balanceamento de classes, se necessário.
- Validação externa em dados reais anonimizados.
"""
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Notebook gerado em: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
