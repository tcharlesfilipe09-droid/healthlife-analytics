"""Executa o pipeline completo: preprocessamento, EDA, visualizações e ML."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analise_exploratoria import executar_eda
from src.machine_learning import treinar_e_comparar
from src.visualizacoes import gerar_todas_visualizacoes


def main() -> None:
    print("=" * 60)
    print("Health&Life Analytics — Pipeline completo")
    print("=" * 60)

    print("\n[1/3] Pré-processamento e EDA...")
    df, estrutura, _stats = executar_eda(usar_processado=True)

    print("\n[2/3] Gerando visualizações...")
    graficos = gerar_todas_visualizacoes(df)

    print("\n[3/3] Treinando e avaliando modelos...")
    comparacao, detalhes = treinar_e_comparar(df)

    print("\n" + "=" * 60)
    print("PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"Registros: {estrutura['n_linhas']} | Colunas: {estrutura['n_colunas']}")
    print(f"Gráficos gerados: {len(graficos)}")
    print("Melhor modelo:", comparacao.iloc[0]["modelo"])
    print("F1-teste:", comparacao.iloc[0]["f1_teste"])
    print("=" * 60)


if __name__ == "__main__":
    main()
