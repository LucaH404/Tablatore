"""Misura i consumi del Tool 3 (trascrivi_note) e prova un agente con i tool 1+2+3.

Uso:  .venv\\Scripts\\python.exe scripts\\misura_tool_trascrizione.py
"""
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
tmp = Path(tempfile.mkdtemp())
AUDIO = ROOT / "tests" / "audio"
proc = psutil.Process()

# --- 1. Consumi ---
from app.tools import trascrizione  # noqa: E402
from app.tools.audio_io import carica_audio, salva_wav  # noqa: E402

ram0 = proc.memory_info().rss
t0 = time.perf_counter()
trascrizione._get_modello()
t_modello = time.perf_counter() - t0

uno_min = np.tile(carica_audio(AUDIO / "solo_chitarra.mp3", 44100), 3)[:, : 60 * 44100]
f60 = salva_wav(tmp / "un_minuto.wav", uno_min, 44100)
t0 = time.perf_counter()
trascrizione.trascrivi_note(f60, cartella_output=tmp)
t_primo = time.perf_counter() - t0
tempi = []
for _ in range(5):
    t0 = time.perf_counter()
    r = trascrizione.trascrivi_note(f60, cartella_output=tmp)
    tempi.append(time.perf_counter() - t0)
print("=== Consumi trascrivi_note (1 minuto di audio, CPU, ONNX) ===")
print(f"caricamento modello {t_modello:.2f} s | prima trascrizione {t_primo:.2f} s (compilazione numba/librosa)")
print(f"successive: media {statistics.mean(tempi):.2f} s (min {min(tempi):.2f}, max {max(tempi):.2f}) | {r.numero_note} note")
print(f"RAM aggiunta dal tool {(proc.memory_info().rss - ram0) / 2**20:.0f} MB | RAM processo {proc.memory_info().rss / 2**20:.0f} MB")

# --- 2. Agente con i tool 1+2+3 ---
os.chdir(tmp)
from datapizza.agents import Agent  # noqa: E402
from datapizza.clients.openai import OpenAIClient  # noqa: E402

from app.tools.separazione import separa_chitarra_audio  # noqa: E402
from app.tools.trascrizione import trascrivi_note_audio  # noqa: E402
from app.tools.validazione import valida_file_audio  # noqa: E402

agente = Agent(
    name="analista_test",
    client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-nano"),
    system_prompt=(
        "Trascrivi brani di chitarra. Procedura: 1) valida_file_audio; se non valido fermati. "
        "2) separa_chitarra_audio; se la chitarra non è presente fermati. "
        "3) trascrivi_note_audio sull''audio_consigliato'; se gli avvisi suggeriscono un'altra "
        "sensibilità riprova una sola volta. Rispondi in una riga: "
        "'FATTO: <numero_note> note da <nota_piu_grave> a <nota_piu_acuta>, file <file_note>' oppure 'STOP: <motivo>'."
    ),
    tools=[valida_file_audio, separa_chitarra_audio, trascrivi_note_audio],
    max_steps=7,
)
print("\n=== Agente con tool 1+2+3 ===")
for f in (AUDIO / "mix.mp3", AUDIO / "solo_chitarra.mp3"):
    t0 = time.perf_counter()
    r = agente.run(f"Trascrivi il file: {f}")
    usati = []
    print(f">>> {f.name}: {r.text.strip()}")
    print(f"    {time.perf_counter() - t0:.2f}s | token in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
