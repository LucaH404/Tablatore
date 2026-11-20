"""Misura i consumi del Tool 1 (valida_file) e verifica che un LLM sappia chiamarlo.

1. Genera un MP3 di 1 minuto, esegue valida_file 20 volte, misura tempo medio e memoria.
2. Crea un agente datapizza-ai minimo con il tool e controlla che lo chiami davvero,
   sia su un file valido sia su un file finto (che deve essere rifiutato).
Uso:  .venv\\Scripts\\python.exe scripts\\misura_tool_validazione.py
"""
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.tools.validazione import valida_file, valida_file_audio  # noqa: E402

load_dotenv(ROOT / ".env")
tmp = Path(tempfile.mkdtemp())
mp3 = tmp / "un_minuto.mp3"
subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=60", "-b:a", "192k", str(mp3)], check=True)

# --- 1. Consumi ---
valida_file(mp3)  # riscaldamento
tempi = []
tracemalloc.start()
for _ in range(20):
    t0 = time.perf_counter()
    valida_file(mp3)
    tempi.append(time.perf_counter() - t0)
_, picco_py = tracemalloc.get_traced_memory()
tracemalloc.stop()

# memoria del sottoprocesso ffprobe (campionata durante l'esecuzione)
proc = subprocess.Popen([shutil.which("ffprobe"), "-v", "error", "-show_format", "-show_streams", str(mp3)],
                        stdout=subprocess.DEVNULL)
picco_ff = 0
try:
    pp = psutil.Process(proc.pid)
    while proc.poll() is None:
        picco_ff = max(picco_ff, pp.memory_info().rss)
except psutil.NoSuchProcess:
    pass
proc.wait()

print("=== Consumi valida_file (MP3 da 1 minuto, 192 kbps) ===")
print(f"tempo medio: {statistics.mean(tempi) * 1000:.0f} ms | min {min(tempi) * 1000:.0f} ms | max {max(tempi) * 1000:.0f} ms")
print(f"memoria Python di picco: {picco_py / 1024:.0f} KB | ffprobe RSS di picco: {picco_ff / 1_048_576:.1f} MB")

# --- 2. Un LLM chiama davvero il tool? ---
from datapizza.agents import Agent  # noqa: E402
from datapizza.clients.openai import OpenAIClient  # noqa: E402

finto = tmp / "finto.mp3"
finto.write_text("non sono audio " * 50)

agente = Agent(
    name="guardiano_test",
    client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-nano"),
    system_prompt="Controlli i file caricati con il tool valida_file_audio. "
                  "Rispondi solo 'ACCETTATO' oppure 'RIFIUTATO: <motivo>'.",
    tools=[valida_file_audio],
    max_steps=3,
)
print("\n=== Chiamata del tool tramite agente datapizza-ai ===")
for f in (mp3, finto):
    t0 = time.perf_counter()
    r = agente.run(f"Controlla il file: {f}")
    print(f"{f.name}: '{r.text.strip()}' | {time.perf_counter() - t0:.2f}s | "
          f"token in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
