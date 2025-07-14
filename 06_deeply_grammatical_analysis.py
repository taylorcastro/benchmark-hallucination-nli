# -*- coding: utf-8 -*-
# 05_enriquecer_y_analizar_gramaticalmente.py

import os
import json
import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Cargar modelo spaCy (ajustar a idioma si necesario)
nlp = spacy.load("en_core_web_sm")  # Cambia a "es_core_news_sm" si tu corpus está en español

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/03_gramatical_2.0")
output_dir = Path("/home/dslab/teilor/atom/data/03_gramatical_2.0/enriched_full")
output_dir.mkdir(parents=True, exist_ok=True)

# Función: métricas lingüísticas extendidas
def enriquecer_df(df):
    for prefijo in ["prompt", "resp"]:
        tokens = df[f"{prefijo}_longitud_tokens"].replace(0, 1)
        df[f"{prefijo}_verbos_por_token"] = df[f"{prefijo}_num_verbos"] / tokens
        df[f"{prefijo}_sustantivos_por_token"] = df[f"{prefijo}_num_sustantivos"] / tokens
        df[f"{prefijo}_pregunta_verbos"] = df[f"{prefijo}_tiene_pregunta"] * df[f"{prefijo}_num_verbos"]
    return df

# Función: calcular profundidad de árbol sintáctico
def calcular_profundidad(doc):
    def profundidad_token(token):
        profundidad = 0
        while token.head != token:
            profundidad += 1
            token = token.head
        return profundidad
    return max((profundidad_token(t) for t in doc), default=0)

# Iterar sobre archivos extendidos
archivos = sorted(input_dir.glob("analisis_gramatical_extend_*.csv"))
for archivo in archivos:
    try:
        df = pd.read_csv(archivo)
        if df.empty or "prompt" not in df.columns or "respuesta" not in df.columns:
            continue
        modelo = df["modelo"].iloc[0] if "modelo" in df.columns else archivo.stem
        tqdm.pandas(desc=f"Procesando {modelo}")

        # Enriquecer con métricas gramaticales
        df = enriquecer_df(df)

        # Enriquecer con profundidad sintáctica
        df["prompt_tree_depth"] = df["prompt"].progress_apply(lambda x: calcular_profundidad(nlp(str(x))))
        df["resp_tree_depth"] = df["respuesta"].progress_apply(lambda x: calcular_profundidad(nlp(str(x))))

        # Guardar
        salida = output_dir / archivo.name.replace("extend_", "extend_enriched_")
        df.to_csv(salida, index=False)
        print(f"Guardado: {salida}")

    except Exception as e:
        print(f"Error procesando {archivo.name}: {e}")
