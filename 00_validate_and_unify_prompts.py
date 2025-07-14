# -*- coding: utf-8 -*-
import os
import pandas as pd
import json

BENCHMARKS_PATH = "/home/dslab/teilor/atom/data/benchmarks"
OUTPUT_JSON = "/home/dslab/teilor/atom/data/prompts_unificados_limpio.json"

def map_original_label_to_type(label, benchmark):
    label = str(label).strip().lower()
    if benchmark == "truthfulqa":
        return "factual"
    elif benchmark == "fever":
        if label in ["supported", "supports"]:
            return "factual"
        if label in ["refuted", "refutes"]:
            return "hallucinated"
        if "not enough info" in label:
            return "uncertain"
    elif benchmark == "halueval":
        if label == "pass":
            return "factual"
        if label == "fail":
            return "hallucinated"
    elif benchmark == "advglue":
        if label in ["1", "entailment"]:
            return "factual"
        if label in ["0", "contradiction"]:
            return "hallucinated"
        if label in ["2", "neutral"]:
            return "uncertain"
    elif benchmark == "dynabench":
        return "factual"
    return "unknown"

def infer_subtype_from_filename(filename):
    filename = filename.lower()
    if "qa" in filename or "truthfulqa" in filename:
        return "qa"
    if "statement" in filename or "claim" in filename or "fever" in filename:
        return "statement"
    return "unknown"

def clasificar_longitud(texto):
    longitud = len(texto)
    if longitud <= 100:
        return "short"
    elif longitud <= 300:
        return "medium"
    else:
        return "long"

all_prompts = []

for benchmark_name in os.listdir(BENCHMARKS_PATH):
    benchmark_dir = os.path.join(BENCHMARKS_PATH, benchmark_name)
    if not os.path.isdir(benchmark_dir):
        continue

    for file_name in os.listdir(benchmark_dir):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(benchmark_dir, file_name)
        try:
            df = pd.read_csv(file_path)

            prompt_col = "prompt" if "prompt" in df.columns else "text"
            if prompt_col not in df.columns:
                print(f"Archivo sin columna de texto valida: {file_path}")
                continue

            df = df.rename(columns={prompt_col: "prompt"})
            df["benchmark"] = benchmark_name

            if "type" not in df.columns:
                df["type"] = None

            if "label" in df.columns:
                df["original_label"] = df["label"]
            elif "original_label" not in df.columns:
                df["original_label"] = df["type"]

            df["original_label"] = df["original_label"].fillna(
                "factual" if benchmark_name == "truthfulqa" else "unknown"
            )

            df["type"] = df.apply(
                lambda row: map_original_label_to_type(row["original_label"], benchmark_name), axis=1
            )

            if "is_question" in df.columns:
                df["subtype"] = df["is_question"].apply(
                    lambda x: "qa" if str(x).strip().lower() == "true" else "statement"
                )
            else:
                df["subtype"] = infer_subtype_from_filename(file_name)

            df["metadata"] = df.apply(
                lambda row: {"source_file": file_name, "subtype": row["subtype"]}, axis=1
            )

            df["length_class"] = df["prompt"].apply(clasificar_longitud)

            df = df[["prompt", "benchmark", "type", "original_label", "metadata", "length_class"]]
            df = df.dropna(subset=["prompt", "type"])
            df = df.drop_duplicates(subset=["prompt"])

            all_prompts.extend(df.to_dict(orient="records"))

        except Exception as e:
            print(f"Error procesando {file_path}: {e}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_prompts, f, indent=2, ensure_ascii=False)

print(f"Total prompts unificados: {len(all_prompts)}")
print(f"Archivo guardado: {OUTPUT_JSON}")
