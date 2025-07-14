# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
import pandas as pd

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/nli/02")
output_dir = Path("/home/dslab/teilor/atom/data/nli/04")
output_dir.mkdir(parents=True, exist_ok=True)

# Umbrales de intensidad
umbral_leve = 33
umbral_media = 66
umbral_nli_contradiction = 0.35  # Umbral mínimo para considerar una contradicción

# Función para obtener la etiqueta más probable
def extraer_etiqueta_predicha(pred_str, threshold=umbral_nli_contradiction):
    try:
        etiquetas = eval(pred_str)
        if isinstance(etiquetas, list):
            for etiqueta in etiquetas:
                if etiqueta.get("label") == "contradiction" and etiqueta.get("score", 0) >= threshold:
                    return "contradiction"
            return etiquetas[0].get("label", "neutral")
    except Exception:
        return "error"
    return "neutral"

# Función auxiliar para análisis por modelo
def calcular_metrica_por_modelo(oraciones, etiquetas):
    total = len(oraciones)
    contradicciones = sum(1 for et in etiquetas if et == "contradiction")
    porcentaje = round(100 * contradicciones / total, 2) if total > 0 else 0.0
    es_alu = contradicciones > 0
    if porcentaje == 0:
        intensidad = "ninguna"
    elif porcentaje <= umbral_leve:
        intensidad = "leve"
    elif porcentaje <= umbral_media:
        intensidad = "media"
    else:
        intensidad = "alta"
    return es_alu, porcentaje, intensidad

# Procesar archivos
archivos = sorted(input_dir.glob("results_*_nli.json"))
for archivo in archivos:
    print(f"Procesando: {archivo.name}")
    with open(archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        oraciones = row.get("oraciones", [])
        for modelo in ["roberta", "deberta", "distilroberta"]:
            columna = f"nli_{modelo}"
            etiquetas_raw = row.get(columna, [])
            etiquetas = [extraer_etiqueta_predicha(p) for p in etiquetas_raw]
            es_alu, porcentaje, intensidad = calcular_metrica_por_modelo(oraciones, etiquetas)
            row[f"label_{modelo}"] = etiquetas
            row[f"es_alucinacion_{modelo}"] = es_alu
            row[f"porcentaje_alucinacion_{modelo}"] = porcentaje
            row[f"intensidad_{modelo}"] = intensidad

    salida = output_dir / archivo.name
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Guardado en: {salida}")
