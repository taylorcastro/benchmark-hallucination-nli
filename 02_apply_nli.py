# -*- coding: utf-8 -*-
import os
import json
import spacy
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline
from collections import Counter
import time

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/respuesta")
output_dir = Path("/home/dslab/teilor/atom/data/nli")
output_dir.mkdir(parents=True, exist_ok=True)

# Modelos NLI
modelos_nli = {
    "roberta-large-mnli": "nli_roberta",
    "microsoft/deberta-large-mnli": "nli_deberta",
    "cross-encoder/nli-distilroberta-base": "nli_distilroberta"
}

# Cargar spaCy
nlp = spacy.load("en_core_web_sm")

# Inicializar clasificadores
clasificadores = {
    nombre_col: pipeline(
        "text-classification",
        model=modelo,
        top_k=None,
        truncation=True,
        max_length=512
    )
    for modelo, nombre_col in modelos_nli.items()
}

def clasificar_con_reintentos(modelo, texto, reintentos=3, espera=2):
    for intento in range(reintentos):
        try:
            salida = modelo(texto)
            if isinstance(salida, list) and isinstance(salida[0], dict) and "label" in salida[0]:
                return salida[0]["label"].lower()
            else:
                return str(salida[0]).lower()
        except Exception as e:
            if intento < reintentos - 1:
                time.sleep(espera)
            else:
                return f"error: {str(e)}"

def analizar_oraciones(row):
    prompt = row["prompt"]
    respuesta = row.get("response", "")
    if respuesta is None or not isinstance(respuesta, str) or not respuesta.strip():
        row = dict(row)
        row["oraciones"] = []
        row["nli_roberta"] = []
        row["nli_deberta"] = []
        row["nli_distilroberta"] = []
        row["nli_consenso"] = []
        row["oraciones_alucinadas"] = []
        row["es_alucinacion"] = False
        row["es_alucinacion_any_nli"] = False
        row["porcentaje_oraciones_alucinadas"] = 0.0
        row["intensidad_alucinacion"] = "no_analizado"
        row["errores_nli"] = {col: 0 for col in modelos_nli.values()}
        row["nli_estado"] = "no_processed"
        return row

    doc = nlp(respuesta)
    oraciones = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    predicciones = {col: [] for col in modelos_nli.values()}
    errores = {col: 0 for col in modelos_nli.values()}

    for i, oracion in enumerate(oraciones):
        texto = f"{prompt} </s></s> {oracion}"
        print(f"  - Oracion {i+1}: {oracion}")
        for columna, modelo in clasificadores.items():
            etiqueta = clasificar_con_reintentos(modelo, texto)
            predicciones[columna].append(etiqueta)
            if "error" in etiqueta:
                errores[columna] += 1
            print(f"    [{columna}] -> {etiqueta}")

    nli_consenso = []
    for i in range(len(oraciones)):
        votos = [
            predicciones["nli_roberta"][i],
            predicciones["nli_deberta"][i],
            predicciones["nli_distilroberta"][i]
        ]
        if any("error" in v for v in votos):
            nli_consenso.append("error")
            continue
        for etiqueta in ["contradiction", "entailment", "neutral"]:
            if votos.count(etiqueta) >= 2:
                nli_consenso.append(etiqueta)
                break
        else:
            nli_consenso.append("ambiguous")

    oraciones_alucinadas = [
        oraciones[i] for i, etiqueta in enumerate(nli_consenso) if etiqueta == "contradiction"
    ]

    n_total = len(oraciones)
    n_alucinadas = len(oraciones_alucinadas)
    porcentaje = n_alucinadas / max(n_total, 1)
    es_alucinacion = n_alucinadas > 0

    hay_alucinacion_individual = any(
        "contradiction" in [predicciones["nli_roberta"][i], predicciones["nli_deberta"][i], predicciones["nli_distilroberta"][i]]
        for i in range(n_total)
    )

    if n_total == 0:
        intensidad = "no_analizado"
    elif porcentaje == 0:
        intensidad = "ninguna"
    elif porcentaje < 0.33:
        intensidad = "leve"
    elif porcentaje < 0.66:
        intensidad = "moderada"
    else:
        intensidad = "grave"

    print(f"  -> Alucinaciones: {n_alucinadas}/{n_total} oraciones ({porcentaje:.2%})")

    row = dict(row)
    row["oraciones"] = oraciones
    row["nli_roberta"] = predicciones["nli_roberta"]
    row["nli_deberta"] = predicciones["nli_deberta"]
    row["nli_distilroberta"] = predicciones["nli_distilroberta"]
    row["nli_consenso"] = nli_consenso
    row["oraciones_alucinadas"] = oraciones_alucinadas
    row["es_alucinacion"] = es_alucinacion
    row["es_alucinacion_any_nli"] = hay_alucinacion_individual
    row["porcentaje_oraciones_alucinadas"] = porcentaje
    row["intensidad_alucinacion"] = intensidad
    row["errores_nli"] = errores
    row["nli_estado"] = "processed"
    return row

if __name__ == "__main__":
    archivos = sorted(input_dir.glob("results_*.json"))
    for archivo in archivos:
        salida = output_dir / archivo.name.replace(".json", "_nli.json")

        if salida.exists():
            print(f"\nYa procesado: {archivo.name}. Saltando.")
            continue

        print(f"\nProcesando: {archivo.name}")
        with open(archivo, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if "prompt" not in df.columns or ("response" not in df.columns and "respuesta" not in df.columns):
            print("  Columnas requeridas no encontradas. Saltando.")
            continue
        if "respuesta" in df.columns and "response" not in df.columns:
            df["response"] = df["respuesta"]

        df["modelo"] = archivo.name.replace("results_", "").replace(".json", "")

        resultados = []
        errores_globales = Counter()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Clasificando oraciones"):
            print(f"\nFila {idx+1}/{len(df)} - Modelo: {row['modelo']}")
            enriched = analizar_oraciones(row.copy())
            resultados.append(enriched)

            for modelo, count in enriched.get("errores_nli", {}).items():
                errores_globales[modelo] += count

            if len(resultados) % 50 == 0:
                with open(salida, "w", encoding="utf-8") as out:
                    json.dump(resultados, out, indent=2, ensure_ascii=False)
                print(f"  Guardado parcial: {len(resultados)} filas")

        with open(salida, "w", encoding="utf-8") as out:
            json.dump(resultados, out, indent=2, ensure_ascii=False)

        print(f"\nGuardado final en: {salida}")
        if any(errores_globales.values()):
            print("Resumen de errores:")
            for modelo, count in errores_globales.items():
                print(f"  - {modelo}: {count} errores")
        else:
            print("Sin errores de inferencia.")
