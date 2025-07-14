# -*- coding: utf-8 -*-

import json
import time
import psutil
import requests
from pathlib import Path
import pandas as pd

# Rutas y configuracion
root_dir = Path("/home/dslab/teilor/atom")
input_file = root_dir / "data" / "prompts_unificados_limpio.json"
output_dir = root_dir / "data" / "respuesta"
log_dir = root_dir / "logs"
output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# "llama3:latest","mistral:7b",  "phi:latest", "gemma:7b",  "qwen:latest", "deepseek-coder:6.7b"

modelos_validos = [
     "llama3:latest",
]

with open(input_file, "r", encoding="utf-8") as f:
    prompts_all = json.load(f)

prompts_df = pd.DataFrame(prompts_all)

def detectar_modelos():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=10) #11434
        disponibles = res.json().get("models", [])
        return [m["name"] for m in disponibles]
    except Exception as e:
        print(f"Error detectando modelos: {e}")
        return []

def hacer_peticion_con_reintentos(payload, modelo, registrar_log):
    for intento in range(5):
        try:
            res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            if res.status_code == 200:
                return res.json().get("response", "")
            else:
                registrar_log(f"[{modelo}] HTTP {res.status_code}")
        except Exception as e:
            registrar_log(f"[{modelo}] Reintento {intento+1}/5 fallido: {e}")
            time.sleep(min(2**intento, 120))
    return None

def procesar_modelo(modelo):
    output_path = output_dir / f"results_{modelo.replace(':', '-')}.json"
    log_path = log_dir / f"log_{modelo.replace(':', '-')}.log"

    def registrar_log(texto):
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {texto}\n")

    existentes = set()
    resultados_previos = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    resultados_previos = data
                    existentes = set((r["prompt"], r["benchmark"]) for r in resultados_previos if "prompt" in r and "benchmark" in r)
                else:
                    registrar_log("Archivo previo invalido, ignorado.")
        except Exception as e:
            registrar_log(f"Error cargando resultados previos: {e}")

    nuevos_resultados = []
    errores = 0
    inicio = time.time()
    ram_inicio = round(psutil.virtual_memory().available / 1024**3, 2)

    print(f"\nProcesando modelo: {modelo}")
    total_prompts = len(prompts_df)

    for i, row in prompts_df.iterrows():
        prompt = row["prompt"]
        benchmark = row["benchmark"]
        tipo = row["type"]
        subtipo = row["metadata"].get("subtype", "unknown")

        clave = (prompt, benchmark)
        if clave in existentes:
            continue

        payload = {
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }

        respuesta = hacer_peticion_con_reintentos(payload, modelo, registrar_log)

        if respuesta is not None:
            print(f"[{modelo}] {benchmark} | Prompt {i+1}/{total_prompts} -> {respuesta[:100]}")
            nuevos_resultados.append({
                "modelo": modelo,
                "prompt": prompt,
                "respuesta": respuesta,
                "benchmark": benchmark,
                "type": tipo,
                "subtype": subtipo
            })
        else:
            print(f"[{modelo}] {benchmark} | Prompt {i+1}/{total_prompts} -> timeout")
            nuevos_resultados.append({
                "modelo": modelo,
                "prompt": prompt,
                "respuesta": None,
                "benchmark": benchmark,
                "type": tipo,
                "subtype": subtipo,
                "error": "timeout"
            })
            errores += 1

        if len(nuevos_resultados) >= 25:
            try:
                resultados_previos.extend(nuevos_resultados)
                resultados_previos = list({(r["prompt"], r["benchmark"]): r for r in resultados_previos}.values())
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(resultados_previos, f, indent=2, ensure_ascii=False)
                print(f"[{modelo}] Guardado parcial: {len(resultados_previos)} respuestas")
                nuevos_resultados.clear()
            except Exception as e:
                registrar_log(f"Error guardando parcial: {e}")

    if nuevos_resultados:
        try:
            resultados_previos.extend(nuevos_resultados)
            resultados_previos = list({(r["prompt"], r["benchmark"]): r for r in resultados_previos}.values())
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(resultados_previos, f, indent=2, ensure_ascii=False)
            print(f"[{modelo}] Guardado final: {len(resultados_previos)} respuestas")
        except Exception as e:
            registrar_log(f"Error guardando final: {e}")

    duracion = round(time.time() - inicio, 2)
    monitor = {
        "modelo": modelo,
        "respuestas_totales": len(resultados_previos),
        "errores": errores,
        "ram_inicio": ram_inicio,
        "ram_final": round(psutil.virtual_memory().available / 1024**3, 2),
        "tiempo_total_s": duracion
    }
    try:
        with open(output_path.parent / f"monitor_{modelo.replace(':', '-')}.json", "w", encoding="utf-8") as f:
            json.dump(monitor, f, indent=2, ensure_ascii=False)
        print(f"[{modelo}] Monitor guardado")
    except Exception as e:
        registrar_log(f"Error guardando monitor: {e}")

if __name__ == "__main__":
    disponibles = detectar_modelos()
    compatibles = [m for m in disponibles if m in modelos_validos]

    print("Modelos detectados:", disponibles)
    print("Modelos compatibles:", compatibles)

    for modelo in compatibles:
        try:
            procesar_modelo(modelo)
        except Exception as e:
            log_path = log_dir / f"log_{modelo.replace(':', '-')}.log"
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error procesando modelo completo: {e}\n")
            print(f"Error procesando {modelo}, ver log.")
