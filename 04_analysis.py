# -*- coding: utf-8 -*-
# 04_analysis.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/nli/04")
output_dir = Path("/home/dslab/teilor/atom/data/analysis/05")
output_dir.mkdir(parents=True, exist_ok=True)

# Cargar archivos enriquecidos
archivos = sorted(input_dir.glob("results_*_nli.json"))
df_total = pd.DataFrame()

for archivo in archivos:
    try:
        df = pd.read_json(archivo)
        if df.empty:
            continue
        modelo = df.get("modelo", [archivo.stem])[0]
        df["modelo"] = modelo
        df_total = pd.concat([df_total, df], ignore_index=True)
    except Exception as e:
        print(f"Error procesando {archivo.name}: {e}")

if df_total.empty:
    print("No se encontraron datos válidos.")
    exit()

# Clasificadores por columnas booleanas
clasificadores = {
    "es_alucinacion_roberta": "roberta",
    "es_alucinacion_deberta": "deberta",
    "es_alucinacion_distilroberta": "distilroberta",
    "es_alucinacion": "consenso"
}

# Análisis por clasificador
for col_alu, nombre in clasificadores.items():
    if col_alu not in df_total.columns:
        continue

    resumen_alu = df_total.groupby("modelo")[col_alu].value_counts().unstack(fill_value=0)
    resumen_alu.to_csv(output_dir / f"resumen_alucinaciones_{nombre}.csv")

    resumen_alu.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title(f"Alucinaciones por modelo ({nombre})")
    plt.ylabel("Cantidad")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"grafico_alucinaciones_{nombre}.png")
    plt.close()

    contradicciones = df_total[df_total[col_alu] == True]
    contradicciones.to_csv(output_dir / f"contradicciones_{nombre}.csv", index=False)

# Comparaciones cruzadas
def comparar(df, col1, col2):
    if col1 in df.columns and col2 in df.columns:
        return (df[col1] == df[col2]).mean()
    return None

comparaciones = []
for modelo in df_total["modelo"].unique():
    df_m = df_total[df_total["modelo"] == modelo]
    fila = {"modelo": modelo, "total": len(df_m)}

    for col_alu in clasificadores:
        if col_alu in df_m.columns:
            fila[f"{col_alu}_porcentaje"] = round(df_m[col_alu].mean() * 100, 2)

    fila["roberta_vs_deberta"] = round(comparar(df_m, "es_alucinacion_roberta", "es_alucinacion_deberta") or 0, 3)
    fila["roberta_vs_consenso"] = round(comparar(df_m, "es_alucinacion_roberta", "es_alucinacion") or 0, 3)
    fila["deberta_vs_consenso"] = round(comparar(df_m, "es_alucinacion_deberta", "es_alucinacion") or 0, 3)

    comparaciones.append(fila)

df_global = pd.DataFrame(comparaciones)
df_global.to_csv(output_dir / "resumen_modelo_global.csv", index=False)

# Gráfico resumen de contradicciones por modelo (consenso)
if "es_alucinacion" in df_total.columns:
    porcentaje_contradicciones = (
        df_total.groupby("modelo")["es_alucinacion"].mean() * 100
    ).sort_values(ascending=False)

    porcentaje_contradicciones.plot(
        kind="barh", figsize=(8, 5), color="firebrick", edgecolor="black"
    )
    plt.xlabel("Porcentaje de contradicciones")
    plt.title("Contradicciones por modelo (consenso NLI)")
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_porcentaje_contradicciones_consenso.png")
    plt.close()

# Agrupaciones adicionales por benchmark, tipo y longitud
agrupadores = ["benchmark", "type", "subtype"]
for columna in agrupadores:
    if columna in df_total.columns:
        for col_alu, nombre in clasificadores.items():
            if col_alu in df_total.columns:
                resumen = df_total.groupby([columna, "modelo"])[col_alu].value_counts().unstack(fill_value=0)
                resumen.to_csv(output_dir / f"{columna}_alucinaciones_{nombre}.csv")

print("\nAnálisis completado. Resultados guardados en:", output_dir)
