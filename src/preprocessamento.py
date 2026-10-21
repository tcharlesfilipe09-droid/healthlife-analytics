"""Limpeza, tratamento e engenharia de atributos do dataset de café e saúde."""

from __future__ import annotations

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, ID_COL


def carregar_dados_brutos(caminho: str | None = None) -> pd.DataFrame:
    """Carrega o CSV bruto e valida a existência do arquivo."""
    path = caminho or DATA_RAW
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {path}. "
            "Verifique se o arquivo synthetic_coffee_health_10000.csv está em data/raw/."
        )
    return pd.read_csv(path)


def criar_faixa_etaria(age: float) -> str:
    """Agrupa idade em faixas interpretáveis para análise e modelagem."""
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza e cria variáveis derivadas sem usar a variável-alvo.

    Motivos das alterações:
    - Remoção de duplicatas evita vieses por contagem repetida.
    - Preenchimento de ausentes numéricos pela mediana e categóricos pela moda
      preserva a distribuição central sem descartar linhas desnecessariamente.
    - Caffeine_per_Cup e Age_Group enriquecem o modelo com hipóteses de domínio.
    """
    dados = df.copy()

    # Remove identificador da etapa de análise tabular exportada (mantém cópia limpa)
    if ID_COL in dados.columns and dados[ID_COL].duplicated().any():
        # IDs duplicados não invalidam o registro, mas sinalizam revisão
        pass

    duplicatas = dados.duplicated().sum()
    if duplicatas > 0:
        dados = dados.drop_duplicates().reset_index(drop=True)

    # Tratamento de ausentes numéricos
    numericas = dados.select_dtypes(include="number").columns.tolist()
    for col in numericas:
        if dados[col].isna().any():
            dados[col] = dados[col].fillna(dados[col].median())

    # Tratamento de ausentes categóricos
    categoricas = dados.select_dtypes(exclude="number").columns.tolist()
    for col in categoricas:
        if dados[col].isna().any():
            moda = dados[col].mode(dropna=True)
            valor = moda.iloc[0] if not moda.empty else "Unknown"
            dados[col] = dados[col].fillna(valor)

    # Variável derivada 1: cafeína média por xícara (evita divisão por zero)
    # Justificativa: captura intensidade da bebida além do volume de consumo.
    ingestao = dados["Coffee_Intake"].clip(lower=0.1)
    dados["Caffeine_per_Cup"] = dados["Caffeine_mg"] / ingestao

    # Variável derivada 2: faixa etária (útil em EDA e como feature categórica)
    dados["Age_Group"] = dados["Age"].apply(criar_faixa_etaria)

    # Variável derivada 3: indicador de hábitos de risco combinados
    # Justificativa: tabaco + álcool podem interagir com sono/estresse.
    dados["Risk_Habits"] = (
        (dados["Smoking"].astype(float) > 0).astype(int)
        + (dados["Alcohol_Consumption"].astype(float) > 0).astype(int)
    )

    return dados


def exportar_processado(df: pd.DataFrame, caminho: str | None = None) -> None:
    """Salva o dataset tratado em CSV."""
    path = caminho or DATA_PROCESSED
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def executar_preprocessamento() -> pd.DataFrame:
    """Pipeline completo de pré-processamento com exportação."""
    bruto = carregar_dados_brutos()
    tratado = tratar_dados(bruto)
    exportar_processado(tratado)
    return tratado


if __name__ == "__main__":
    df_final = executar_preprocessamento()
    print(f"Dataset processado: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")
    print(f"Salvo em: {DATA_PROCESSED}")
