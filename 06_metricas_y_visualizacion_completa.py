# -*- coding: utf-8 -*-
"""
06_metricas_y_visualizacion_completa_EXTENDIDO.py

Incluye cálculo de Kappa, visualizaciones ampliadas, y test de significancia Mann-Whitney.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score
)
from scipy.stats import mannwhitneyu

# === CONFIGURACIÓN ===
data_dir = Path("/home/dslab/teilor/atom/data")
input_gram_dir = data_dir / "03_gramatical_2.0/enriched_full_with_benchmark"
kappa_input = data_dir / "05/df_total_unificado.csv"
output_dir = data_dir / "04_analysis_2.0"
output_dir.mkdir(parents=True, exist_ok=True)

clasificadores = {
    "es_alucinacion_roberta": "roberta",
    "es_alucinacion_deberta": "deberta",
    "es_alucinacion_distilroberta": "distilroberta",
    "es_alucinacion": "consenso"
}

label_map = {
    "llama3-latest": "llama3:8b",
    "mistral-7b": "mistral0.2:7b",
    "phi-latest": "phi3:3.8b",
    "gemma-7b": "gemma3:4b",
    "qwen-latest": "qwen2.5:7b",
    "deepseek-coder-6.7b": "deepseek-coder\n6.7b"
}

# === 1. LEER DATOS ===
gram_files = sorted(input_gram_dir.glob("analisis_gramatical_extend_enriched_*.csv"))
df_gram = pd.concat([pd.read_csv(f) for f in gram_files], ignore_index=True)
df_gram["modelo"] = df_gram["modelo"].replace(label_map)

# === 2. CALCULAR MÉTRICAS ===
metricas_totales, ranking_alucinacion = [], []
metricas_por_benchmark = []
mannwhitney_results = []

variables_linguisticas = [
    "longitud_tokens", 
    "longitud_caracteres", 
    "num_verbos", 
    "num_sustantivos", 
    "num_adjetivos", 
    "num_adverbios"
]

for modelo, df in df_gram.groupby("modelo"):
    if "gold_binaria" in df.columns and df["gold_binaria"].dtype == object:
        df["gold_binaria"] = df["gold_binaria"].map({"non_hallucinated": 0, "hallucinated": 1})

    df_eval = df[df["gold_binaria"].isin([0,1])]
    if not df_eval.empty:
        y_true, y_pred = df_eval["gold_binaria"], df_eval["es_alucinacion"]
        metricas_totales.append({
            "modelo": modelo,
            "accuracy": round(accuracy_score(y_true,y_pred),4),
            "precision": round(precision_score(y_true,y_pred, zero_division=0),4),
            "recall": round(recall_score(y_true,y_pred, zero_division=0),4),
            "f1_score": round(f1_score(y_true,y_pred, zero_division=0),4),
            "total": len(df_eval)
        })

    for bmk, sub in df.groupby("benchmark"):
        y_true, y_pred = sub["gold_binaria"], sub["es_alucinacion"]
        if y_true.notna().sum() > 0:
            metricas_por_benchmark.append({
                "modelo": modelo,
                "benchmark": bmk,
                "accuracy": round(accuracy_score(y_true,y_pred),4),
                "precision": round(precision_score(y_true,y_pred, zero_division=0),4),
                "recall": round(recall_score(y_true,y_pred, zero_division=0),4),
                "f1_score": round(f1_score(y_true,y_pred, zero_division=0),4),
                "total": len(sub)
            })

        # === TEST MANN-WHITNEY ===
        for var in variables_linguisticas:
            if var in sub.columns:
                grupo_a = sub[sub["es_alucinacion"]==1][var].dropna()
                grupo_b = sub[sub["es_alucinacion"]==0][var].dropna()
                if len(grupo_a) > 0 and len(grupo_b) > 0:
                    stat, p = mannwhitneyu(grupo_a, grupo_b, alternative='two-sided')
                    mannwhitney_results.append({
                        "modelo": modelo,
                        "benchmark": bmk,
                        "variable": var,
                        "grupo_a_media": grupo_a.mean(),
                        "grupo_b_media": grupo_b.mean(),
                        "U_statistic": stat,
                        "p_value": round(p,5),
                        "significativo": p < 0.05
                    })

    ranking_alucinacion.append({
        "modelo": modelo,
        "porcentaje_alucinacion": round(df["es_alucinacion"].mean()*100, 2)
    })

# === 3. HEATMAP DATA ===
df_heatmap = df_gram.groupby(['modelo', 'benchmark'])['es_alucinacion'].mean().reset_index()
df_heatmap['% Hallucinations'] = df_heatmap['es_alucinacion'] * 100

# === 4. COHEN'S KAPPA ===
df_total = pd.read_csv(kappa_input)
cols = ['modelo', 'es_alucinacion_roberta', 'es_alucinacion_deberta', 'es_alucinacion_distilroberta']
df_kappa_calc = df_total[cols].copy()

for col in ['es_alucinacion_roberta', 'es_alucinacion_deberta', 'es_alucinacion_distilroberta']:
    df_kappa_calc[col] = df_kappa_calc[col].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)

kappa_rows = []
range_kappa_rows = []

for modelo, sub in df_kappa_calc.groupby("modelo"):
    pares = [
        ("es_alucinacion_roberta", "es_alucinacion_distilroberta"),
        ("es_alucinacion_roberta", "es_alucinacion_deberta"),
        ("es_alucinacion_distilroberta", "es_alucinacion_deberta")
    ]
    kappas = []
    for a, b in pares:
        mask = sub[a].notna() & sub[b].notna()
        if mask.sum() > 0 and len(set(sub.loc[mask,a])) > 1 and len(set(sub.loc[mask,b])) > 1:
            kappa = cohen_kappa_score(sub.loc[mask,a], sub.loc[mask,b])
            kappas.append(kappa)
            kappa_rows.append({
                "modelo": modelo,
                "comparacion": f"{a.replace('es_alucinacion_','').capitalize()} vs {b.replace('es_alucinacion_','').capitalize()}",
                "kappa": round(kappa,3)
            })
    if kappas:
        range_kappa_rows.append({
            "modelo": modelo,
            "max_kappa": round(max(kappas),3),
            "min_kappa": round(min(kappas),3)
        })

# === 5. GUARDAR RESULTADOS ===
pd.DataFrame(metricas_totales).to_csv(output_dir/"metricas_modelos.csv", index=False)
pd.DataFrame(metricas_por_benchmark).to_csv(output_dir/"metricas_por_benchmark.csv", index=False)
pd.DataFrame(ranking_alucinacion).to_csv(output_dir/"ranking_porcentaje_alucinacion.csv", index=False)
pd.DataFrame(kappa_rows).to_csv(output_dir/"cohen_kappa_por_modelo.csv", index=False)
pd.DataFrame(range_kappa_rows).to_csv(output_dir/"cohen_kappa_rango.csv", index=False)
pd.DataFrame(mannwhitney_results).to_csv(output_dir/"mannwhitney_test_results.csv", index=False)
df_heatmap.to_csv(output_dir/"heatmap_data.csv", index=False)

# === 6. GRAFICAS ===
plt.figure(figsize=(12,6))
heatmap_pivot = df_heatmap.pivot(index="modelo", columns="benchmark", values="% Hallucinations")
sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': '% Hallucinations'}, annot_kws={"size":14})
plt.title("Hallucination Heatmap by Model and Benchmark", fontsize=16)
plt.xlabel("Benchmark", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir/"heatmap_modelo_benchmark.png")

pivot_kappa = pd.DataFrame(kappa_rows).pivot(index="modelo", columns="comparacion", values="kappa")
plt.figure(figsize=(10,6))
sns.heatmap(pivot_kappa, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size":14})
plt.title("Cohen's Kappa by Model and Comparison", fontsize=16)
plt.xlabel("Comparison", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir/"cohen_kappa_heatmap.png")

plt.figure(figsize=(10,6))
sns.barplot(data=pd.DataFrame(kappa_rows), x="kappa", y="comparacion", hue="modelo")
plt.title("Cohen's Kappa Barplot", fontsize=16)
plt.xlabel("Kappa", fontsize=14)
plt.ylabel("Comparison", fontsize=14)
plt.legend(title="Model", fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir/"cohen_kappa_barplot.png")

# === 2.1 TEST MANN-WHITNEY POR MODELO ===
# Input: todos los archivos *_gold.csv en /03_gramatical_2.0/
input_gram_gold_dir = data_dir / "03_gramatical_2.0"
gram_gold_files = sorted(input_gram_gold_dir.glob("analisis_gramatical_extend_*_gold.csv"))

df_gram_gold = pd.concat([pd.read_csv(f) for f in gram_gold_files], ignore_index=True)
df_gram_gold["modelo"] = df_gram_gold["modelo"].replace(label_map)

variables_linguisticas_gold = [
    "longitud_tokens", 
    "longitud_caracteres", 
    "num_verbos", 
    "num_sustantivos", 
    "num_adjetivos", 
    "num_adverbios"
]

mannwhitney_por_modelo = []

for modelo, df in df_gram_gold.groupby("modelo"):
    for var in variables_linguisticas_gold:
        if var in df.columns:
            grupo_a = df[df["es_alucinacion"] == 1][var].dropna()
            grupo_b = df[df["es_alucinacion"] == 0][var].dropna()

            if len(grupo_a) > 0 and len(grupo_b) > 0:
                stat, p = mannwhitneyu(grupo_a, grupo_b, alternative='two-sided')
                mannwhitney_por_modelo.append({
                    "modelo": modelo,
                    "variable": var,
                    "grupo_alucinado_media": round(grupo_a.mean(), 3),
                    "grupo_no_alucinado_media": round(grupo_b.mean(), 3),
                    "U_statistic": round(stat, 3),
                    "p_value": round(p, 5),
                    "significativo": p < 0.05
                })

pd.DataFrame(mannwhitney_por_modelo).to_csv(output_dir / "mannwhitney_por_modelo.csv", index=False)

print("Script ejecutado. Resultados y gráficas guardadas en:", output_dir)
