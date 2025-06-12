from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time
import json
import platform
import os

# === Carga de modelo y procesador ===
print("Cargando modelo...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

# === Selección de dispositivo ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print("Dispositivo:", device)
print("CPU:", platform.processor())
print("Torch:", torch.__version__)
print("Modelo dtype:", model.dtype)

# === Imagen de entrada ===
image_path = "test.jpg"
print(f"\nCargando imagen: {image_path}")
raw_image = Image.open(image_path).convert("RGB")
inputs = processor(images=raw_image, return_tensors="pt").to(device)

# === Caption base (por defecto) ===
print("\nGenerando caption base (sin configuraciones adicionales)...")
with torch.no_grad():
    base_output = model.generate(**inputs)
base_caption = processor.decode(base_output[0], skip_special_tokens=True)
print("Caption base:", base_caption)

# === Configuraciones a probar ===
param_grid = [
    {"min_length": 10},
    {"min_length": 30},
    {"min_length": 50}
]

# === Carpeta de salida ===
os.makedirs("resultados", exist_ok=True)

# === Evaluación por configuración ===
for config_id, params in enumerate(param_grid, start=1):
    print(f"\n==============================")
    print(f" Configuración {config_id}: {params}")
    print(f"==============================")

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_length=params["min_length"],
            max_length=120,
            length_penalty=1.2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_beams=7,
            return_dict_in_generate=True,
            early_stopping=False,
            output_scores=True
        )
    end_time = time.time()
    inference_time = end_time - start_time

    # === Resultados resumidos ===
    print("\nResumen:\n")
    results = []
    for i, (seq, score) in enumerate(zip(output.sequences, output.sequences_scores), start=1):
        caption = processor.decode(seq, skip_special_tokens=True)
        delta_vs_base = len(set(caption.split()) ^ set(base_caption.split()))
        print(f"Score: {score.item():.4f} | Palabras distintas vs base: {delta_vs_base}")
        print(f"\n{caption}")
        results.append({
            "caption": caption,
            "score": score.item(),
            "tokens": processor.tokenizer.convert_ids_to_tokens(seq.tolist()),
            "config": params,
            "inference_time": round(inference_time, 2),
            "delta_vs_base": delta_vs_base
        })

    # === Scores paso a paso por token (solo guardado en archivo) ===
    step_scores = []
    for step, score_tensor in enumerate(output.scores, start=1):
        top_values, top_indices = torch.topk(score_tensor[0], k=5)
        decoded_tokens = [processor.tokenizer.decode([i]) for i in top_indices]
        step_scores.append({
            "step": step,
            "top_tokens": list(zip(decoded_tokens, [round(v.item(), 2) for v in top_values]))
        })

    # === Guardar JSON detallado ===
    data = {
        "caption_base": base_caption,
        "config": params,
        "results": results,
        "step_scores": step_scores,
        "inference_time": round(inference_time, 2)
    }
    file_name = f"resultados/captions_analysis_{config_id}.json"
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en {file_name}")
    print(f"Inference Time: {inference_time:.2f} seconds\n")
