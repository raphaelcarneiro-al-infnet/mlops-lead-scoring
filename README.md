# Lead Scoring — Pós-Graduação (MLOps)

Projeto de operacionalização de modelo de machine learning para previsão de matrículas em cursos de pós-graduação.

## Problema de Negócio
Prever a probabilidade de um lead se matricular em um curso de pós-graduação, permitindo que o time de Marketing priorize contatos com maior potencial de conversão no HubSpot.

## Estrutura do Projeto

```
lead-scoring-mlops/
├── configs/
│   ├── data.yaml
│   └── pipeline.yaml
├── data/raw/
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── dimensionality.py
│   ├── evaluate.py
│   └── monitoring.py
├── api/
│   └── app.py
├── models/
├── outputs/
├── mlruns/
├── notebooks/
├── .gitignore
├── README.md
└── requirements.txt
```

## Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar os modelos
```bash
python src/train.py
```

### 3. Visualizar experimentos no MLflow
```bash
mlflow ui --backend-store-uri mlruns
```
Acesse http://localhost:5000

### 4. Redução de dimensionalidade
```bash
python src/dimensionality.py
```

### 5. Subir a API de inferência
```bash
uvicorn api.app:app --reload
```
Acesse http://127.0.0.1:8000/docs

### 6. Monitoramento
```bash
python src/monitoring.py
```

## Resultados

| Modelo | F1 | Recall | CV F1 |
|---|---|---|---|
| Random Forest | 0.3737 | 0.7701 | 0.3677 |
| XGBoost | 0.3698 | 0.7452 | 0.3624 |
| Perceptron | 0.3170 | 0.6620 | 0.2199 |
| Decision Tree | 0.2761 | 0.7812 | 0.2852 |

**Modelo escolhido: Random Forest** — melhor F1 e CV F1, com Recall de 77%.

## Tecnologias
Python 3.11 | scikit-learn | MLflow | FastAPI | XGBoost | pandas