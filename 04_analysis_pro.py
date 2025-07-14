# -*- coding: utf-8 -*-
# 04_analysis_pro.py
# Punto de partida de gráficos

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/nli/04")
output_dir = Path("/home/dslab/teilor/atom/data/analysis/05")
output_dir.mkdir(parents=True, exist_ok=True)

# Cargar archivos enriquecidos
archivos = sorted(input_dir.glob("results_*_nli.json"))
df_total = pd.DataFrame()

for archivo in archivos:
    try:
        with open(archivo, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            continue
        df = pd.DataFrame(data)
        modelo = df.get("modelo", [archivo.stem])[0]
        df["modelo"] = modelo

        # Recalculo de es_alucinacion por consenso (mayoria >= 2)
        def consenso(row):
            votos = sum([
                bool(row.get("es_alucinacion_roberta")),
                bool(row.get("es_alucinacion_deberta")),
                bool(row.get("es_alucinacion_distilroberta"))
            ])
            return votos >= 2

        df["es_alucinacion"] = df.apply(consenso, axis=1)
        df_total = pd.concat([df_total, df], ignore_index=True)
    except Exception as e:
        print(f"Error procesando {archivo.name}: {e}")

if df_total.empty:
    print("No se encontraron datos validos.")
    exit()

# Clasificadores por columnas booleanas
clasificadores = {
    "es_alucinacion_roberta": "roberta",
    "es_alucinacion_deberta": "deberta",
    "es_alucinacion_distilroberta": "distilroberta",
    "es_alucinacion": "consenso"
}

# Clasificacion de longitud de respuesta
if "respuesta" in df_total.columns:
    def clasificar_longitud(texto):
        if not isinstance(texto, str):
            return "desconocida"
        longitud = len(texto.split())
        if longitud < 15:
            return "corta"
        elif longitud < 50:
            return "media"
        else:
            return "larga"

    df_total["length_class"] = df_total["respuesta"].apply(clasificar_longitud)
    df_total = df_total[df_total["length_class"] != "desconocida"]

# Analisis por clasificador
for col_alu, nombre in clasificadores.items():
    if col_alu not in df_total.columns:
        continue

    resumen_alu = df_total.groupby("modelo")[col_alu].value_counts().unstack(fill_value=0)
    resumen_alu.to_csv(output_dir / f"resumen_alucinaciones_{nombre}.csv")

    resumen_alu.plot(kind="bar", stacked=True, figsize=(10, 5), color=["red", "green"])
    plt.title(f"Alucinaciones por modelo ({nombre})")
    plt.ylabel("Cantidad")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"grafico_alucinaciones_{nombre}.png")
    plt.close()

    porcentaje_alu = df_total.groupby("modelo")[col_alu].mean() * 100
    porcentaje_alu.plot(kind="bar", figsize=(10, 5))
    plt.title(f"Porcentaje de alucinaciones por modelo ({nombre})")
    plt.ylabel("Porcentaje")
    plt.xlabel("Modelo")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"grafico_porcentaje_alucinaciones_{nombre}.png")
    plt.close()

    contradicciones = df_total[df_total[col_alu] == True]
    contradicciones.to_csv(output_dir / f"contradicciones_{nombre}.csv", index=False)

    if "length_class" in df_total.columns:
        resumen_length = df_total.groupby(["modelo", "length_class"])[col_alu].mean().unstack().fillna(0) * 100
        resumen_length.to_csv(output_dir / f"length_class_alucinaciones_{nombre}.csv")

        resumen_length.plot(kind="bar", figsize=(10, 5))
        plt.title(f"Porcentaje de alucinaciones por longitud y modelo ({nombre})")
        plt.ylabel("Porcentaje de alucinaciones")
        plt.xlabel("Modelo")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="length_class")
        plt.tight_layout()
        plt.savefig(output_dir / f"grafico_porcentaje_alucinaciones_longitud_{nombre}.png")
        plt.close()

# Comparaciones cruzadas entre clasificadores
comparaciones = []
for modelo in df_total["modelo"].unique():
    df_m = df_total[df_total["modelo"] == modelo]
    fila = {"modelo": modelo, "total": len(df_m)}
    for col_alu in clasificadores:
        if col_alu in df_m.columns:
            fila[f"{col_alu}_porcentaje"] = round(df_m[col_alu].mean() * 100, 2)
    fila["roberta_vs_deberta"] = round((df_m["es_alucinacion_roberta"] == df_m["es_alucinacion_deberta"]).mean() * 100, 2)
    fila["roberta_vs_consenso"] = round((df_m["es_alucinacion_roberta"] == df_m["es_alucinacion"]).mean() * 100, 2)
    fila["deberta_vs_consenso"] = round((df_m["es_alucinacion_deberta"] == df_m["es_alucinacion"]).mean() * 100, 2)
    comparaciones.append(fila)

pd.DataFrame(comparaciones).to_csv(output_dir / "resumen_modelo_global.csv", index=False)

# Comparativa general por modelo y clasificador (agrupado)
pct = defaultdict(list)
modelos = sorted(df_total["modelo"].unique())
for nombre in clasificadores.values():
    for modelo in modelos:
        col = f"es_alucinacion_{nombre}" if nombre != "consenso" else "es_alucinacion"
        pct[nombre].append(df_total[df_total.modelo == modelo][col].mean() * 100)

pct_df = pd.DataFrame(pct, index=modelos)
pct_df.plot(kind="bar", figsize=(12, 6))
plt.title("Porcentaje de alucinaciones por modelo y clasificador NLI")
plt.ylabel("Porcentaje")
plt.xlabel("modelo")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "grafico_porcentaje_alucinaciones_nli.png")
plt.close()

# Agrupaciones adicionales por benchmark, type, subtype y length_class
agrupadores = ["benchmark", "type", "subtype", "length_class"]
for columna in agrupadores:
    if columna in df_total.columns:
        for col_alu, nombre in clasificadores.items():
            if col_alu in df_total.columns:
                resumen = df_total.groupby([columna, "modelo"])[col_alu].value_counts().unstack(fill_value=0)
                resumen.to_csv(output_dir / f"{columna}_alucinaciones_{nombre}.csv")
                
# Grafico por longitud
for col_alu, nombre in clasificadores.items():
    if col_alu not in df_total.columns:
        continue
    resumen_length = (
        df_total.groupby(["modelo", "length_class"])[col_alu].mean()
        .unstack()
        .fillna(0)
        * 100
    )
    resumen_length.to_csv(output_dir / f"longitud_alucinaciones_{nombre}.csv")
    resumen_length.plot(kind="bar")
    plt.title(f"Porcentaje de alucinaciones por longitud y modelo ({nombre})")
    plt.ylabel("Porcentaje de alucinaciones")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(output_dir / f"grafico_porcentaje_alucinaciones_longitud_{nombre}.png")
    plt.close()

# ================================
# Gráfico 4: Heatmap por modelo y benchmark (consenso)
# ================================
if "benchmark" in df_total.columns and "es_alucinacion" in df_total.columns:
    heatmap_data = (
        df_total.groupby(["modelo", "benchmark"])["es_alucinacion"]
        .mean()
        .unstack()
        .fillna(0) * 100
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Heatmap de alucinaciones por modelo y benchmark (consenso)")
    plt.xlabel("Benchmark")
    plt.ylabel("Modelo")
    plt.tight_layout()
    plt.savefig(output_dir / "grafico_4_heatmap_modelo_benchmark_consenso.png")
    plt.close()


print("\nAnalisis completado. Resultados guardados en:", output_dir)

