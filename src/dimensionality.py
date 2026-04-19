import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analisar_variancia_pca(X_train_proc, n_components=30):
    """Analisa quantos componentes explicam 95% da variância."""
    pca_analise = PCA(n_components=min(n_components, X_train_proc.shape[1]))
    pca_analise.fit(X_train_proc)

    variancia_acumulada = np.cumsum(pca_analise.explained_variance_ratio_)
    n_components_95 = np.argmax(variancia_acumulada >= 0.95) + 1

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(variancia_acumulada) + 1), variancia_acumulada, marker='o', markersize=3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variância')
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} componentes')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('PCA — Variância Explicada Acumulada')
    plt.legend()
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/pca_variancia.png', bbox_inches='tight')
    plt.close()
    print(f"Componentes para 95% da variância: {n_components_95}")
    return n_components_95

def treinar_com_reducao(
    nome_experimento: str,
    redutor,
    X_train_proc, X_test_proc,
    y_train, y_test,
    config_pipeline: dict
):
    """Treina Random Forest após redução de dimensionalidade e registra no MLflow."""
    cfg_mlflow = config_pipeline['mlflow']
    cfg_rf = config_pipeline['models']['random_forest']

    mlflow.set_tracking_uri(cfg_mlflow['tracking_uri'])
    mlflow.set_experiment(cfg_mlflow['experiment_name'])

    with mlflow.start_run(run_name=nome_experimento):
        # Aplica redução
        if redutor is not None:
            X_train_red = redutor.fit_transform(X_train_proc, y_train)
            X_test_red = redutor.transform(X_test_proc)
            n_dims_orig = X_train_proc.shape[1]
            n_dims_red = X_train_red.shape[1]
        else:
            X_train_red = X_train_proc
            X_test_red = X_test_proc
            n_dims_orig = X_train_proc.shape[1]
            n_dims_red = X_train_proc.shape[1]

        print(f"\n{'='*50}")
        print(f"Experimento: {nome_experimento}")
        print(f"Dimensões: {n_dims_orig} → {n_dims_red}")

        # Treina RF nas dimensões reduzidas
        rf = RandomForestClassifier(
            n_estimators=cfg_rf['n_estimators'],
            max_depth=cfg_rf['max_depth'],
            min_samples_leaf=cfg_rf['min_samples_leaf'],
            random_state=cfg_rf['random_state'],
            class_weight=cfg_rf['class_weight'],
            n_jobs=-1
        )
        rf.fit(X_train_red, y_train)
        y_pred = rf.predict(X_test_red)

        # Métricas
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        cv_scores = cross_val_score(rf, X_train_red, y_train, cv=5, scoring='f1')

        # Loga no MLflow
        mlflow.log_param("tecnica_reducao", nome_experimento)
        mlflow.log_param("dimensoes_originais", n_dims_orig)
        mlflow.log_param("dimensoes_reduzidas", n_dims_red)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())

        print(classification_report(y_test, y_pred))
        print(f"Dimensões: {n_dims_orig} → {n_dims_red}")
        print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return {
            'f1': f1, 'precision': precision, 'recall': recall,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
            'dims_orig': n_dims_orig, 'dims_red': n_dims_red
        }

def plotar_lda_2d(X_train_proc, y_train):
    """Visualiza separação das classes no espaço LDA."""
    lda_viz = LDA(n_components=1)
    X_lda = lda_viz.fit_transform(X_train_proc, y_train)

    plt.figure(figsize=(10, 4))
    for classe, label, cor in [(0, 'Não Matriculou', 'steelblue'), (1, 'Matriculou', 'darkorange')]:
        plt.hist(X_lda[y_train == classe, 0], bins=50, alpha=0.6, label=label, color=cor)
    plt.xlabel('Componente LDA 1')
    plt.ylabel('Frequência')
    plt.title('LDA — Separação das Classes')
    plt.legend()
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/lda_separacao.png', bbox_inches='tight')
    plt.close()
    print("Gráfico LDA salvo em outputs/lda_separacao.png")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from ingest import ingerir
    from preprocess import criar_preprocessador, dividir_dados

    config_data = carregar_config('configs/data.yaml')
    config_pipeline = carregar_config('configs/pipeline.yaml')

    # Carrega e prepara dados
    df = ingerir()
    X_train, X_test, y_train, y_test = dividir_dados(df, config_data)
    preprocessador = criar_preprocessador(config_data)

    # Aplica preprocessamento (fit no treino, transform em ambos)
    X_train_proc = preprocessador.fit_transform(X_train)
    X_test_proc = preprocessador.transform(X_test)

    print(f"\nDimensões após preprocessamento: {X_train_proc.shape[1]} features")

    # Analisa PCA
    n_components_95 = analisar_variancia_pca(X_train_proc)

    # Experimento 1: Sem redução (baseline)
    r_baseline = treinar_com_reducao(
        "RF_sem_reducao",
        None,
        X_train_proc, X_test_proc,
        y_train, y_test,
        config_pipeline
    )

    # Experimento 2: PCA (95% variância)
    pca = PCA(n_components=n_components_95, random_state=42)
    r_pca = treinar_com_reducao(
        f"RF_PCA_{n_components_95}componentes",
        pca,
        X_train_proc, X_test_proc,
        y_train, y_test,
        config_pipeline
    )

    # Experimento 3: LDA (máximo n_classes - 1 = 1 componente)
    lda = LDA(n_components=1)
    r_lda = treinar_com_reducao(
        "RF_LDA_1componente",
        lda,
        X_train_proc, X_test_proc,
        y_train, y_test,
        config_pipeline
    )

    # Visualização LDA
    plotar_lda_2d(X_train_proc, y_train)

    # Resumo comparativo
    print("\n\n=== RESUMO — IMPACTO DA REDUÇÃO DE DIMENSIONALIDADE ===")
    print(f"{'Técnica':<30} {'Dims':>6} {'F1':>6} {'Recall':>8} {'Precision':>10} {'CV F1':>8}")
    print("-" * 72)
    for nome, r in [("Sem redução", r_baseline), (f"PCA ({r_pca['dims_red']} comp.)", r_pca), ("LDA (1 comp.)", r_lda)]:
        print(f"{nome:<30} {r['dims_red']:>6} {r['f1']:>6.4f} {r['recall']:>8.4f} {r['precision']:>10.4f} {r['cv_mean']:>8.4f}")

    # Salva gráfico comparativo
    tecnicas = ['Sem redução', f"PCA ({r_pca['dims_red']} comp.)", 'LDA (1 comp.)']
    f1s = [r_baseline['f1'], r_pca['f1'], r_lda['f1']]
    recalls = [r_baseline['recall'], r_pca['recall'], r_lda['recall']]

    x = np.arange(len(tecnicas))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, f1s, width, label='F1-Score', color='steelblue')
    ax.bar(x + width/2, recalls, width, label='Recall', color='darkorange')
    ax.set_ylabel('Score')
    ax.set_title('Impacto da Redução de Dimensionalidade')
    ax.set_xticks(x)
    ax.set_xticklabels(tecnicas)
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/comparacao_reducao.png', bbox_inches='tight')
    plt.close()
    print("\nGráfico salvo em outputs/comparacao_reducao.png")