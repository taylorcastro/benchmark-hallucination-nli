# -*- coding: utf-8 -*-
# generate_prompts_unificados.py

import pandas as pd
import os

# Directorio de entrada
input_benchmarks_dir = "/home/dslab/teilor/atom/data/benchmarks/"
output_path = "/home/dslab/teilor/atom/data/prompts_unificados_limpio.json"

# Cargar los benchmarks
truthfulqa = pd.read_json(os.path.join(input_benchmarks_dir, "truthfulqa_prompts.json"), lines=True)
dynabench = pd.read_json(os.path.join(input_benchmarks_dir, "dynabench.json"), lines=True)

# Unificar los prompts
truthfulqa['benchmark'] = 'truthfulqa'
dynabench['benchmark'] = 'dynabench'

# Seleccionar columnas comunes
cols = ['prompt', 'benchmark', 'type']
all_prompts = pd.concat([truthfulqa[cols], dynabench[cols]], ignore_index=True)

# Limpieza y eliminación de duplicados
all_prompts.drop_duplicates(subset=['prompt'], inplace=True)

# Guardar como JSONL
all_prompts.to_json(output_path, orient='records', lines=True)

print(f"Archivo generado: {output_path}")
