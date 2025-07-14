# -*- coding: utf-8 -*-
"""
06_metricas_y_visualizacion_completa.py (versión final corregida)

- Métricas de clasificación con conversión gold_binaria
- Cohen’s Kappa con saneo de booleanos y validación robusta
- Visualización de boxplots por modelo y alucinación
"""

import json, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score
)
from scipy.stats import mannwhitneyu
import numpy as np

# === CONFIGURACIÓN ===
data_dir        = Path("/home/dslab/teilor/atom/data")
input_gram_dir  = data_dir / "03_gramatical_2.0/enriched_full"
input_nli_dir   = data_dir / "02_nli/05"
output_dir      = data_dir / "04_analysis_2.0"
output_dir.mkdir(parents=True, exist_ok=True)

clasificadores = {
    "es_alucinacion_roberta"      : "roberta",
    "es_alucinacion_deberta"      : "deberta",
    "es_alucinacion_distilroberta": "distilroberta",
    "es_alucinacion"              : "consenso"
}

vars_linguisticas = [
    "prompt_longitud_tokens","prompt_num_verbos","prompt_num_sustantivos",
    "prompt_num_adjetivos","prompt_num_adverbios","prompt_longitud_chars",
    "resp_longitud_tokens","resp_num_verbos","resp_num_sustantivos",
    "resp_num_adjetivos","resp_num_adverbios","resp_longitud_chars"
]
resp_cols = [v for v in vars_linguisticas if v.startswith("resp_")]

# === 1. LECTURA Y PROCESADO DE ARCHIVOS CSV ===
metricas_totales, ranking_alucinacion = [], []
resumenes_linguisticos, significancia_test = [], []
archivos = sorted(input_gram_dir.glob("analisis_gramatical_extend_enriched_*.csv"))
df_todos = []

for archivo in archivos:
    df = pd.read_csv(archivo)
    modelo = df.get("modelo", pd.Series([archivo.stem])).iloc[0]

    for col in resp_cols:
        if col not in df.columns:
            df[col] = 0
    df[resp_cols] = df[resp_cols].fillna(0)
    df["modelo"] = modelo

    if "gold_binaria" in df.columns and df["gold_binaria"].dtype == object:
        df["gold_binaria"] = df["gold_binaria"].map({
            "non_hallucinated": 0, "hallucinated": 1
        })

    df_todos.append(df)

    if {"gold_binaria", "es_alucinacion"}.issubset(df.columns):
        df_eval = df[df["gold_binaria"].isin([0,1])]
        if not df_eval.empty:
            y_true, y_pred = df_eval["gold_binaria"], df_eval["es_alucinacion"]
            metricas_totales.append({
                "modelo"   : modelo,
                "accuracy" : round(accuracy_score(y_true,y_pred),4),
                "precision": round(precision_score(y_true,y_pred),4),
                "recall"   : round(recall_score(y_true,y_pred),4),
                "f1_score" : round(f1_score(y_true,y_pred),4),
                "total"    : len(df_eval)
            })
            pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T.to_csv(
                output_dir / f"classification_report_{modelo}.csv"
            )

    if "es_alucinacion" in df.columns:
        ranking_alucinacion.append({
            "modelo": modelo,
            "porcentaje_alucinacion": round(df["es_alucinacion"].mean()*100, 2)
        })

    resumen = df.groupby("es_alucinacion").agg({v:["mean"] for v in vars_linguisticas})
    resumen.columns = ["_".join(c) for c in resumen.columns]
    resumen["modelo"] = modelo
    resumen = resumen.reset_index()
    resumenes_linguisticos.append(resumen)

    for var in vars_linguisticas:
        if var in df.columns:
            g0 = df[df.es_alucinacion==0][var]
            g1 = df[df.es_alucinacion==1][var]
            if len(g0) > 0 and len(g1) > 0:
                _, p = mannwhitneyu(g0, g1, alternative="two-sided")
                significancia_test.append({
                    "modelo": modelo, "variable": var,
                    "p_value": round(p, 5), "significativo": p < 0.05
                })

# === 2. CONCATENAR TODOS LOS MODELOS ===
df_all = pd.concat(df_todos, ignore_index=True) if df_todos else pd.DataFrame()

# === 3. COHEN’S KAPPA ===
kappa_rows = []
for modelo, sub in df_all.groupby("modelo"):
    # Asegurarse de que estén en tipo bool, incluso si vienen como strings
    for col in clasificadores:
        if col in sub.columns:
            sub[col] = sub[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    pares = [
        ("es_alucinacion_roberta", "es_alucinacion_distilroberta"),
        ("es_alucinacion_roberta", "es_alucinacion_deberta"),
        ("es_alucinacion_distilroberta", "es_alucinacion_deberta")
    ]
    for a, b in pares:
        if {a,b}.issubset(sub.columns):
            mask = sub[a].notna() & sub[b].notna()
            if mask.sum() == 0:
                continue
            kappa = cohen_kappa_score(sub.loc[mask,a], sub.loc[mask,b])
            if not np.isnan(kappa):
                kappa_rows.append({
                    "modelo"     : modelo,
                    "comparacion": f"{a}_vs_{b}",
                    "kappa"      : round(kappa,3)
                })

pd.DataFrame(kappa_rows).to_csv(output_dir/"cohen_kappa_por_modelo.csv", index=False)

# === 4. GUARDAR OTROS CSV ===
pd.DataFrame(metricas_totales).to_csv(output_dir/"metricas_modelos.csv", index=False)
pd.DataFrame(ranking_alucinacion).to_csv(output_dir/"ranking_porcentaje_alucinacion.csv", index=False)
pd.concat(resumenes_linguisticos, ignore_index=True).to_csv(output_dir/"resumen_linguistico.csv", index=False)
pd.DataFrame(significancia_test).to_csv(output_dir/"test_significancia_mannwhitney.csv", index=False)

# === 5. GRAFICOS ===
sns.set(style="whitegrid")
for var in ["resp_num_verbos", "resp_num_sustantivos", "resp_longitud_tokens"]:
    if var in df_all.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_all, x="modelo", y=var, hue="es_alucinacion")
        plt.title(f"Distribucion de {var} por modelo y alucinacion")
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{var}_por_modelo.png")
        plt.close()

print("Script final ejecutado correctamente. Resultados guardados en:", output_dir)
