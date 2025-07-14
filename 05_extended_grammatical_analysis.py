# -*- coding: utf-8 -*-
import os
import json
import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Cargar modelo spaCy en inglés
nlp = spacy.load("en_core_web_sm")

# Rutas
input_dir = Path("/home/dslab/teilor/atom/data/02_nli/05")  # Ajusta si es necesario
output_dir = Path("/home/dslab/teilor/atom/data/03_gramatical_2.0")
output_dir.mkdir(parents=True, exist_ok=True)

def extraer_features(texto):
    if not texto or not isinstance(texto, str) or not texto.strip():
        return {
            "longitud_tokens": None,
            "longitud_chars": None,
            "num_verbos": None,
            "num_sustantivos": None,
            "num_adjetivos": None,
            "num_adverbios": None,
            "tiene_pregunta": None,
            "tiene_exclamacion": None
        }

    doc = nlp(texto)
    return {
        "longitud_tokens": len(doc),
        "longitud_chars": len(texto),
        "num_verbos": sum(1 for token in doc if token.pos_ == "VERB"),
        "num_sustantivos": sum(1 for token in doc if token.pos_ == "NOUN"),
        "num_adjetivos": sum(1 for token in doc if token.pos_ == "ADJ"),
        "num_adverbios": sum(1 for token in doc if token.pos_ == "ADV"),
        "tiene_pregunta": int("?" in texto),
        "tiene_exclamacion": int("!" in texto)
    }

# Procesamiento por archivo
archivos = sorted(input_dir.glob("results_*_nli_gold.json"))
if not archivos:
    print("No se encontraron archivos de entrada.")
    exit()

for archivo in archivos:
    modelo = archivo.stem.replace("results_", "").replace("_nli", "")
    output_file = output_dir / f"analisis_gramatical_extend_{modelo}.csv"

    if output_file.exists():
        print(f"[Omitido] Ya existe: {output_file}")
        continue

    try:
        with open(archivo, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error leyendo {archivo.name}: {e}")
        continue

    if not data:
        print(f"[Vacio] {archivo.name}")
        continue

    modelo_detectado = data[0].get("modelo", modelo)
    registros = []

    for fila in tqdm(data, desc=f"Procesando {modelo_detectado}"):
        prompt = fila.get("prompt", "")
        respuesta = fila.get("respuesta", "")
        gold = fila.get("gold_binaria", None)
        es_alu = fila.get("es_alucinacion", None)

        feat_prompt = extraer_features(prompt)
        feat_resp = extraer_features(respuesta)

        registro = {
            "modelo": modelo_detectado,
            "prompt": prompt,
            "respuesta": respuesta,
            "gold_binaria": gold,
            "es_alucinacion": es_alu
        }
        registro.update({f"prompt_{k}": v for k, v in feat_prompt.items()})
        registro.update({f"resp_{k}": v for k, v in feat_resp.items()})
        registros.append(registro)

    if registros:
        df_out = pd.DataFrame(registros)
        df_out.to_csv(output_file, index=False)
        print(f"[Guardado] {output_file}")
    else:
        print(f"[Aviso] No hay registros validos para {modelo}")
