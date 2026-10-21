"""Análise exploratória de dados (EDA) e exportação de estatísticas."""

from __future__ import annotations

import pandas as pd

from .config import ESTATISTICAS_PATH, NUMERIC_COLS
from .preprocessamento import carregar_dados_brutos, executar_preprocessamento


def resumo_estrutura(df: pd.DataFrame) -> dict:
    """Retorna métricas básicas de estrutura do DataFrame."""
    return {
        "n_linhas": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "colunas": list(df.columns),
        "tipos": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "ausentes": df.isna().sum().to_dict(),
        "duplicados": int(df.duplicated().sum()),
    }


def detectar_outliers_iqr(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Conta outliers pelo método IQR (1.5 * IQR) para colunas numéricas."""
    cols = colunas or [c for c in NUMERIC_COLS if c in df.columns]
    registros = []
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        n_out = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())
        registros.append(
            {
                "variavel": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": lim_inf,
                "limite_superior": lim_sup,
                "n_outliers": n_out,
                "pct_outliers": round(100 * n_out / len(df), 2),
            }
        )
    return pd.DataFrame(registros)


def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas ampliadas das variáveis numéricas."""
    num = df.select_dtypes(include="number")
    desc = num.describe().T
    desc["median"] = num.median()
    desc["missing"] = num.isna().sum()
    return desc.reset_index().rename(columns={"index": "variavel"})


def perfil_categoricas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Frequências das variáveis categóricas."""
    cats = df.select_dtypes(exclude="number")
    return {col: df[col].value_counts(dropna=False) for col in cats.columns}


def exportar_estatisticas(df: pd.DataFrame, caminho=None) -> pd.DataFrame:
    """Exporta estatísticas descritivas para CSV."""
    path = caminho or ESTATISTICAS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = estatisticas_descritivas(df)
    stats.to_csv(path, index=False)
    return stats


def executar_eda(usar_processado: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Executa EDA e exporta estatísticas."""
    if usar_processado:
        df = executar_preprocessamento()
    else:
        df = carregar_dados_brutos()

    estrutura = resumo_estrutura(df)
    outliers = detectar_outliers_iqr(df)
    stats = exportar_estatisticas(df)

    print("=== Estrutura ===")
    print(f"Linhas: {estrutura['n_linhas']} | Colunas: {estrutura['n_colunas']}")
    print(f"Duplicados: {estrutura['duplicados']}")
    print("Ausentes:", estrutura["ausentes"])
    print("\n=== Outliers (IQR) ===")
    print(outliers.to_string(index=False))
    print(f"\nEstatísticas exportadas em: {ESTATISTICAS_PATH}")
    return df, estrutura, stats


if __name__ == "__main__":
    executar_eda()
