import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, roc_curve, auc)

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plotar_matriz_confusao(y_test, y_pred, nome_modelo: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão — {nome_modelo}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/confusao_{nome_modelo}.png', bbox_inches='tight')
    plt.show()
    print(f"Salvo em outputs/confusao_{nome_modelo}.png")

def comparar_modelos(resultados: dict):
    df = pd.DataFrame(resultados).T
    df = df.sort_values('f1', ascending=False)
    print("\n=== COMPARAÇÃO FINAL DOS MODELOS ===")
    print(df.to_string())

    plt.figure(figsize=(10, 5))
    df[['f1', 'recall', 'precision']].plot(kind='bar', figsize=(10, 5))
    plt.title('Comparação de Métricas por Modelo')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/comparacao_modelos.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from ingest import ingerir
    from preprocess import dividir_dados

    config_data = carregar_config('configs/data.yaml')
    df = ingerir()
    _, X_test, _, y_test = dividir_dados(df, config_data)

    modelos_salvos = ['perceptron', 'decision_tree', 'random_forest', 'xgboost']
    resultados = {}

    for nome in modelos_salvos:
        caminho = f'models/{nome}.pkl'
        if os.path.exists(caminho):
            pipeline = joblib.load(caminho)
            y_pred = pipeline.predict(X_test)
            plotar_matriz_confusao(y_test, y_pred, nome)
            resultados[nome] = {
                'f1': f1_score(y_test, y_pred),
                'recall': __import__('sklearn.metrics', fromlist=['recall_score']).recall_score(y_test, y_pred),
                'precision': __import__('sklearn.metrics', fromlist=['precision_score']).precision_score(y_test, y_pred, zero_division=0)
            }

    comparar_modelos(resultados)