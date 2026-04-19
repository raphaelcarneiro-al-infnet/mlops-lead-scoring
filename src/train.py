import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (classification_report, f1_score,
                              precision_score, recall_score, accuracy_score)
from xgboost import XGBClassifier

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def obter_modelos(config_pipeline: dict) -> dict:
    cfg = config_pipeline['models']
    return {
        'perceptron': Perceptron(
            max_iter=cfg['perceptron']['max_iter'],
            random_state=cfg['perceptron']['random_state'],
            class_weight=cfg['perceptron']['class_weight']
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=cfg['decision_tree']['max_depth'],
            min_samples_leaf=cfg['decision_tree']['min_samples_leaf'],
            min_samples_split=cfg['decision_tree']['min_samples_split'],
            random_state=cfg['decision_tree']['random_state'],
            class_weight=cfg['decision_tree']['class_weight']
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=cfg['random_forest']['n_estimators'],
            max_depth=cfg['random_forest']['max_depth'],
            min_samples_leaf=cfg['random_forest']['min_samples_leaf'],
            random_state=cfg['random_forest']['random_state'],
            class_weight=cfg['random_forest']['class_weight'],
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=cfg['xgboost']['n_estimators'],
            max_depth=cfg['xgboost']['max_depth'],
            learning_rate=cfg['xgboost']['learning_rate'],
            random_state=cfg['xgboost']['random_state'],
            scale_pos_weight=cfg['xgboost']['scale_pos_weight'],
            eval_metric='logloss',
            verbosity=0
        )
    }

def treinar_e_registrar(
    nome_modelo: str,
    pipeline: Pipeline,
    X_train, X_test, y_train, y_test,
    config_pipeline: dict,
    config_data: dict
):
    cfg_mlflow = config_pipeline['mlflow']
    cfg_cv = config_pipeline['cross_validation']

    mlflow.set_tracking_uri(cfg_mlflow['tracking_uri'])
    mlflow.set_experiment(cfg_mlflow['experiment_name'])

    with mlflow.start_run(run_name=nome_modelo):
        print(f"\n{'='*50}")
        print(f"Treinando: {nome_modelo}")

        # Treina
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Métricas
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Validação cruzada
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cfg_cv['folds'],
            scoring=cfg_cv['scoring']
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Loga parâmetros
        mlflow.log_param("modelo", nome_modelo)
        mlflow.log_param("test_size", config_data['dataset']['test_size'])
        mlflow.log_param("cv_folds", cfg_cv['folds'])
        mlflow.log_params(pipeline.named_steps['modelo'].get_params())

        # Loga métricas
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_f1_mean", cv_mean)
        mlflow.log_metric("cv_f1_std", cv_std)

        # Loga modelo
        mlflow.sklearn.log_model(pipeline, "model")

        print(classification_report(y_test, y_pred))
        print(f"CV F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Salva localmente
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, f'models/{nome_modelo}.pkl')
        print(f"Modelo salvo em models/{nome_modelo}.pkl")

        return pipeline, {
            'f1': f1, 'precision': precision,
            'recall': recall, 'accuracy': accuracy,
            'cv_mean': cv_mean, 'cv_std': cv_std
        }

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from ingest import ingerir
    from preprocess import criar_preprocessador, dividir_dados

    config_data = carregar_config('configs/data.yaml')
    config_pipeline = carregar_config('configs/pipeline.yaml')

    df = ingerir()
    X_train, X_test, y_train, y_test = dividir_dados(df, config_data)
    preprocessador = criar_preprocessador(config_data)
    modelos = obter_modelos(config_pipeline)

    resultados = {}
    for nome, modelo in modelos.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessador),
            ('modelo', modelo)
        ])
        _, metricas = treinar_e_registrar(
            nome, pipeline,
            X_train, X_test, y_train, y_test,
            config_pipeline, config_data
        )
        resultados[nome] = metricas

    print("\n\n=== RESUMO COMPARATIVO ===")
    print(f"{'Modelo':<20} {'F1':>6} {'Recall':>8} {'Precision':>10} {'CV F1':>8}")
    print("-" * 55)
    for nome, m in resultados.items():
        print(f"{nome:<20} {m['f1']:>6.4f} {m['recall']:>8.4f} {m['precision']:>10.4f} {m['cv_mean']:>8.4f}")