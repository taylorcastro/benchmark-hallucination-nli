# -*- coding: utf-8 -*-
# 02_apply_nli_generacion.py

import pandas as pd
import json
import glob
import os
from transformers import pipeline
from tqdm import tqdm

# Directorios de entrada
monitor_dir = "/home/dslab/teilor/atom/data/01_respuesta/"
prompts_path = "/home/dslab/teilor/atom/data/prompts_unificados_limpio.json"
output_path = "/home/dslab/teilor/atom/data/unificados_etiquetados_gold.json"

# Cargar prompts
df_prompts = pd.read_json(prompts_path, lines=True)

# Cargar respuestas generadas
monitor_files = glob.glob(os.path.join(monitor_dir, "monitor_*.json"))
all_data = []
for file in monitor_files:
    df = pd.read_json(file, lines=True)
    all_data.append(df)

df_monitor = pd.concat(all_data, ignore_index=True)

# Preparar pipeline NLI
nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=0)

# Aplicar NLI y generar etiquetas
results = []
for idx, row in tqdm(df_monitor.iterrows(), total=len(df_monitor)):
    prompt = row['prompt']
    modelo = row['modelo']
    respuesta = row['response']

    # Aquí se asume respuesta ya segmentada por oraciones (simplificación)
    oraciones = respuesta.split('.')

    orac_alucinadas = 0
    nli_labels = []
    for oracion in oraciones:
        if len(oracion.strip()) == 0:
            continue
        output = nli_pipeline(oracion, prompt, truncation=True)
        label = output[0]['label']
        nli_labels.append(label)
        if label == 'contradiction':
            orac_alucinadas += 1

    porcentaje_alucinadas = orac_alucinadas / len(oraciones)

    # Buscar si tiene gold
    match = df_prompts[df_prompts['prompt'] == prompt]
    if not match.empty and 'gold' in match.columns:
        gold_label = match['gold'].values[0]
        if gold_label in [0, 1]:
            gold_binaria = gold_label
        else:
            gold_binaria = None
    else:
        gold_binaria = None

    # Seudogold por consenso si no hay gold
    seudogold = 1 if orac_alucinadas > 0 else 0

    # Guardar resultado
    results.append({
        'prompt': prompt,
        'modelo': modelo,
        'nli_labels': nli_labels,
        'porcentaje_oraciones_alucinadas': porcentaje_alucinadas,
        'es_alucinacion': seudogold if gold_binaria is None else None,
        'gold_binaria': gold_binaria
    })

# Guardar como JSONL
with open(output_path, 'w') as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Archivo generado: {output_path}")
