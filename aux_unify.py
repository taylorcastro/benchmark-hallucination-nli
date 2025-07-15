# -*- coding: utf-8 -*-import pandas as pd
# 11_unificacion_llm.py

import pandas as pd
import glob
import os

# Directorios base
base_path = "/home/dslab/teilor/atom/data/"

# Cargar etiquetas NLI y consenso
df_labels = pd.read_json(os.path.join(base_path, "unificados_etiquetados_gold.json"), lines=True)

# Cargar información de los prompts
df_prompts = pd.read_json(os.path.join(base_path, "prompts_unificados_limpio.json"), lines=True)

# Cargar respuestas generadas por los modelos (monitor_*.json)
monitor_files = glob.glob(os.path.join(base_path, "01_respuesta/monitor_*.json"))
df_monitor_list = [pd.read_json(file, lines=True) for file in monitor_files]
df_monitor = pd.concat(df_monitor_list, ignore_index=True)

# Realizar merge de los datasets
# Primero: etiquetas NLI + metadata de prompts
df = df_labels.merge(df_prompts, on="prompt", how="left", suffixes=("", "_prompt"))

# Segundo: añadir respuestas de los modelos
df = df.merge(df_monitor[['prompt', 'modelo', 'response']], on=["prompt", "modelo"], how="left")

# Renombrar por claridad
df.rename(columns={"response": "respuesta"}, inplace=True)

# Exportar archivo final
output_path = os.path.join(base_path, "df_total_unificado.csv")
df.to_csv(output_path, index=False)

print(f"Archivo generado correctamente en: {output_path}")
