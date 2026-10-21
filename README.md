# Health&Life Analytics — Café, Cafeína e Saúde

Projeto acadêmico de análise de dados e Machine Learning para a empresa fictícia **Health&Life Analytics**.

O objetivo é investigar relações entre consumo de café, ingestão de cafeína, qualidade do sono, estresse e indicadores de saúde, além de construir modelos preditivos para a variável `Sleep_Quality`.

> A base utilizada é **sintética**. Os resultados descrevem padrões observados nos dados e **não** devem ser interpretados como prova de causalidade clínica.

## Objetivos

- Realizar análise exploratória completa (EDA).
- Limpar e preparar os dados com transparência metodológica.
- Produzir visualizações e insights comparativos.
- Treinar e comparar modelos de classificação.
- Exportar resultados reprodutíveis para avaliação acadêmica e publicação no GitHub.

## Tecnologias

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Dataset

Arquivo: `data/raw/synthetic_coffee_health_10000.csv`

Aproximadamente **10.000 registros** e **16 colunas**:

`ID`, `Age`, `Gender`, `Country`, `Coffee_Intake`, `Caffeine_mg`, `Sleep_Hours`, `Sleep_Quality`, `BMI`, `Heart_Rate`, `Stress_Level`, `Physical_Activity_Hours`, `Health_Issues`, `Occupation`, `Smoking`, `Alcohol_Consumption`

Variáveis derivadas no pré-processamento:

- `Caffeine_per_Cup`
- `Age_Group`
- `Risk_Habits`

## Metodologia

1. **EDA:** inspeção de tipos, ausentes, duplicatas, estatísticas e outliers (IQR).
2. **Tratamento:** imputação simples e engenharia de atributos sem uso do alvo.
3. **Visualização:** histogramas, boxplots, barras, pizza, scatter e heatmaps.
4. **Machine Learning:** classificação de `Sleep_Quality` com Pipeline anti-vazamento.
5. **Comparação:** Random Forest vs Logistic Regression (accuracy, precision, recall, F1, CV, matriz de confusão).
6. **Exportação:** CSV processado, estatísticas, comparação de modelos, gráficos PNG e modelo Joblib.

## Principais descobertas

Resultados obtidos na execução real do pipeline (base sintética de 10.000 registros):

- Correlação `Coffee_Intake` × `Sleep_Hours` ≈ **-0.19**; `Caffeine_mg` × `Sleep_Hours` ≈ **-0.19** (associação linear negativa fraca).
- Consumo médio de café por `Sleep_Quality`: Poor 2.98 · Fair 2.76 · Good 2.45 · Excellent 2.05.
- Horas médias de sono por estresse: High 4.45 · Medium 5.58 · Low 7.25.
- Consumo médio de café por gênero é semelhante (Male ≈ 2.52, Female ≈ 2.50, Other ≈ 2.49).
- `Sleep_Hours` é preditor muito forte de `Sleep_Quality` nesta base sintética — por isso os modelos atingem ~99% de F1; isso **não** implica generalização clínica.

## Resultados dos modelos

| Modelo | Accuracy (teste) | Precision | Recall | F1 (teste) | F1 CV (média) |
|---|---:|---:|---:|---:|---:|
| **Random Forest** (melhor) | 0.9910 | 0.9910 | 0.9910 | **0.9909** | 0.9917 |
| Logistic Regression | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9916 |

Artefatos:

- `results/comparacao_modelos.csv`
- `results/melhor_modelo.joblib`
- matrizes de confusão em `results/graficos/`

O melhor modelo é escolhido automaticamente pelo **F1-score ponderado no conjunto de teste**.

## Instalação e execução

```bash
cd healthlife_analytics
python -m pip install -r requirements.txt
```

### Pipeline completo (recomendado)

```bash
python run_pipeline.py
```

### Notebook

```bash
python scripts/gerar_notebook.py
jupyter nbconvert --to notebook --execute notebooks/analise_completa.ipynb --output analise_completa.ipynb
```

Ou abra `notebooks/analise_completa.ipynb` no Jupyter e execute todas as células.

### Módulos individuais

```bash
python -m src.preprocessamento
python -m src.analise_exploratoria
python -m src.visualizacoes
python -m src.machine_learning
```

## Estrutura de pastas

```text
healthlife_analytics/
├── data/
│   ├── raw/
│   │   └── synthetic_coffee_health_10000.csv
│   └── processed/
│       └── coffee_health_processed.csv
├── notebooks/
│   └── analise_completa.ipynb
├── src/
│   ├── preprocessamento.py
│   ├── analise_exploratoria.py
│   ├── visualizacoes.py
│   └── machine_learning.py
├── results/
│   ├── graficos/
│   ├── estatisticas.csv
│   ├── comparacao_modelos.csv
│   └── melhor_modelo.joblib
├── scripts/
│   └── gerar_notebook.py
├── run_pipeline.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Conclusão

Este repositório demonstra um fluxo acadêmico completo de análise de dados: da exploração à modelagem preditiva, com documentação, reprodutibilidade e exportação de artefatos. As recomendações à Health&Life Analytics devem ser lidas como hipóteses analíticas baseadas em dados sintéticos, sujeitas a validação com dados reais e revisão de especialistas.

## Publicação no GitHub

Comandos sugeridos (após autenticação local):

```bash
cd healthlife_analytics
git init
git add .
git commit -m "Projeto acadêmico Health&Life Analytics: EDA, visualizações e ML"
git branch -M main
git remote add origin https://github.com/<seu-usuario>/<seu-repositorio>.git
git push -u origin main
```

Não versione credenciais, arquivos `.env` ou ambientes virtuais.
