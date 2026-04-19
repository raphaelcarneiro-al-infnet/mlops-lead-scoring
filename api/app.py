from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Lead Scoring API — Pós-Graduação",
    description="API de inferência para previsão de matrícula em pós-graduação.",
    version="1.0.0"
)

# Carrega o modelo vencedor
MODEL_PATH = "models/random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}. Rode src/train.py primeiro.")

pipeline = joblib.load(MODEL_PATH)
print(f"Modelo carregado: {MODEL_PATH}")

class Lead(BaseModel):
    emails_abertos: float = 0.0
    emails_clicados: float = 0.0
    numero_sessoes: float = 0.0
    visualizacoes_pagina: float = 0.0
    fonte_original: Optional[str] = "Não informado"
    cod_programa: Optional[str] = "PGL"
    estado_regiao: Optional[str] = "Não informado"
    area_formacao: Optional[str] = "Não informado"
    area_trabalho: Optional[str] = "Não informado"
    persona: Optional[str] = "Não informado"
    cargo: Optional[str] = "Outros"

class Predicao(BaseModel):
    probabilidade_matricula: float
    classificacao: str
    score_lead: str
    features_recebidas: dict

def classificar_score(prob: float) -> str:
    if prob >= 0.7:
        return "QUENTE 🔥"
    elif prob >= 0.4:
        return "MORNO ⚡"
    else:
        return "FRIO ❄️"

@app.get("/")
def root():
    return {"status": "online", "modelo": "Random Forest — Lead Scoring Pós-Graduação"}

@app.get("/health")
def health():
    return {"status": "healthy", "modelo_carregado": True}

@app.post("/predict", response_model=Predicao)
def predict(lead: Lead):
    try:
        dados = pd.DataFrame([{
            "E-mails de marketing abertos": lead.emails_abertos,
            "E-mails de marketing clicados": lead.emails_clicados,
            "Número de sessões": lead.numero_sessoes,
            "Número de visualizações de página": lead.visualizacoes_pagina,
            "Fonte original": lead.fonte_original,
            "CodPrograma": lead.cod_programa,
            "Estado/Região": lead.estado_regiao,
            "Área de formação": lead.area_formacao,
            "Área de trabalho": lead.area_trabalho,
            "Persona": lead.persona,
            "Cargo": lead.cargo
        }])

        prob = pipeline.predict_proba(dados)[0][1]
        classificacao = int(pipeline.predict(dados)[0])
        score = classificar_score(prob)

        return Predicao(
            probabilidade_matricula=round(float(prob), 4),
            classificacao="Vai matricular" if classificacao == 1 else "Não vai matricular",
            score_lead=score,
            features_recebidas=lead.dict()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modelo/info")
def modelo_info():
    return {
        "algoritmo": "Random Forest",
        "features": [
            "E-mails de marketing abertos",
            "E-mails de marketing clicados",
            "Número de sessões",
            "Número de visualizações de página",
            "Fonte original", "CodPrograma",
            "Estado/Região", "Área de formação",
            "Área de trabalho", "Persona", "Cargo"
        ],
        "metrica_principal": "Recall (foco em encontrar quem vai matricular)",
        "recall": 0.7701,
        "f1_score": 0.3737,
        "threshold": "50% probabilidade"
    }