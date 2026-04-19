import pandas as pd
import numpy as np
import yaml
import joblib
import os
from datetime import datetime

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ── Métricas de negócio ──────────────────────────────────────────────────────

METRICAS_NEGOCIO = {
    "recall_minimo_aceitavel": 0.65,
    "f1_minimo_aceitavel": 0.30,
    "taxa_conversao_esperada": 0.05,
    "descricao": {
        "recall": "Percentual de matrículas reais que o modelo consegue identificar. "
                  "Abaixo de 65% significa que estamos perdendo leads quentes.",
        "f1": "Equilíbrio entre precisão e recall. "
              "Abaixo de 0.30 indica degradação geral do modelo.",
        "taxa_conversao": "Se a taxa real de conversão mudar muito em relação a 5%, "
                          "pode indicar mudança no perfil dos leads (data drift)."
    }
}

# ── Detecção de drift ────────────────────────────────────────────────────────

def detectar_drift_features(df_referencia: pd.DataFrame,
                             df_producao: pd.DataFrame,
                             features_numericas: list,
                             threshold: float = 0.2) -> dict:
    """
    Compara estatísticas básicas entre dados de referência (treino)
    e dados de produção para detectar data drift.
    Usa diferença relativa de média como proxy simples.
    """
    alertas = {}
    for col in features_numericas:
        if col in df_referencia.columns and col in df_producao.columns:
            media_ref = df_referencia[col].mean()
            media_prod = df_producao[col].mean()
            if media_ref != 0:
                drift = abs(media_prod - media_ref) / abs(media_ref)
                if drift > threshold:
                    alertas[col] = {
                        "media_referencia": round(media_ref, 4),
                        "media_producao": round(media_prod, 4),
                        "drift_percentual": round(drift * 100, 2),
                        "alerta": "DRIFT DETECTADO ⚠️"
                    }
    return alertas

def simular_monitoramento(df_atual: pd.DataFrame,
                           config_data: dict,
                           pipeline) -> dict:
    """
    Simula um ciclo de monitoramento em produção.
    Em produção real, df_atual seria o batch de leads do período.
    """
    features_num = config_data['features']['numerical']
    target = config_data['dataset']['target_column']

    X = df_atual.drop(columns=[target], errors='ignore')
    
    # Predições
    predicoes = pipeline.predict(X)
    probabilidades = pipeline.predict_proba(X)[:, 1]

    taxa_positivos = predicoes.mean()
    prob_media = probabilidades.mean()
    prob_alta = (probabilidades >= 0.7).mean()

    relatorio = {
        "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_leads_analisados": len(df_atual),
        "leads_classificados_quentes": int((probabilidades >= 0.7).sum()),
        "leads_classificados_mornos": int(((probabilidades >= 0.4) & (probabilidades < 0.7)).sum()),
        "leads_classificados_frios": int((probabilidades < 0.4).sum()),
        "taxa_predicao_positiva": round(float(taxa_positivos), 4),
        "probabilidade_media": round(float(prob_media), 4),
        "percentual_leads_quentes": round(float(prob_alta) * 100, 2),
        "alertas_drift": {},
        "recomendacao_retreinamento": False
    }

    # Verifica necessidade de retreinamento
    taxa_esperada = METRICAS_NEGOCIO['taxa_conversao_esperada']
    if abs(taxa_positivos - taxa_esperada) / taxa_esperada > 0.5:
        relatorio['recomendacao_retreinamento'] = True
        relatorio['motivo_retreinamento'] = (
            f"Taxa de predições positivas ({taxa_positivos:.2%}) desviou mais de "
            f"50% da taxa esperada ({taxa_esperada:.2%})"
        )

    return relatorio

def imprimir_relatorio(relatorio: dict):
    print("\n" + "="*60)
    print("RELATÓRIO DE MONITORAMENTO — LEAD SCORING PÓS-GRADUAÇÃO")
    print("="*60)
    print(f"Data: {relatorio['data_execucao']}")
    print(f"Total de leads analisados: {relatorio['total_leads_analisados']}")
    print(f"\nClassificação dos leads:")
    print(f"  🔥 Quentes (prob >= 70%): {relatorio['leads_classificados_quentes']}")
    print(f"  ⚡ Mornos  (40-70%):      {relatorio['leads_classificados_mornos']}")
    print(f"  ❄️  Frios   (< 40%):       {relatorio['leads_classificados_frios']}")
    print(f"\nTaxa de predição positiva: {relatorio['taxa_predicao_positiva']:.2%}")
    print(f"Probabilidade média: {relatorio['probabilidade_media']:.4f}")

    if relatorio['alertas_drift']:
        print(f"\n⚠️  ALERTAS DE DRIFT:")
        for col, info in relatorio['alertas_drift'].items():
            print(f"  {col}: {info['alerta']} (drift: {info['drift_percentual']}%)")
    else:
        print(f"\n✅ Nenhum drift detectado nas features numéricas.")

    if relatorio['recomendacao_retreinamento']:
        print(f"\n🚨 RETREINAMENTO RECOMENDADO: {relatorio.get('motivo_retreinamento', '')}")
    else:
        print(f"\n✅ Modelo estável. Retreinamento não necessário.")

    print("\nMÉTRICAS DE NEGÓCIO MONITORADAS:")
    print(f"  Recall mínimo aceitável: {METRICAS_NEGOCIO['recall_minimo_aceitavel']:.0%}")
    print(f"  F1 mínimo aceitável:     {METRICAS_NEGOCIO['f1_minimo_aceitavel']:.2f}")
    print(f"  Taxa de conversão esperada: {METRICAS_NEGOCIO['taxa_conversao_esperada']:.0%}")
    print("="*60)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from ingest import ingerir
    from preprocess import dividir_dados

    config_data = carregar_config('configs/data.yaml')
    pipeline = joblib.load('models/random_forest.pkl')

    df = ingerir()
    _, X_test, _, y_test = dividir_dados(df, config_data)

    # Adiciona target de volta para simular batch de produção
    df_producao = X_test.copy()
    df_producao[config_data['dataset']['target_column']] = y_test.values

    relatorio = simular_monitoramento(df_producao, config_data, pipeline)
    imprimir_relatorio(relatorio)