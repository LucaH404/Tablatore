"""Misura i consumi del Tool 4 (analizza_ritmo) su GPU e CPU e prova un agente con i tool 1-4.

Uso:  .venv\\Scripts\\python.exe scripts\\misura_tool_ritmo.py
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
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
tmp = Path(tempfile.mkdtemp())
AUDIO = ROOT / "tests" / "audio"
proc = psutil.Process()

from app.tools import ritmo  # noqa: E402
from app.tools.audio_io import carica_audio, salva_wav  # noqa: E402

uno_min = np.tile(carica_audio(AUDIO / "mix.mp3", 44100), 3)[:, : 60 * 44100]
f60 = salva_wav(tmp / "un_minuto.wav", uno_min, 44100)


def misura(dispositivo):
    ritmo._rilevatore = None
    ram0 = proc.memory_info().rss
    if dispositivo == "cpu":
        torch.cuda.is_available = lambda: False          # forza la CPU
    else:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ritmo._get_rilevatore()
    t_modello = time.perf_counter() - t0
    t0 = time.perf_counter()
    ritmo.analizza_ritmo(f60, tmp)
    t_primo = time.perf_counter() - t0
    tempi = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = ritmo.analizza_ritmo(f60, tmp)
        tempi.append(time.perf_counter() - t0)
    print(f"=== analizza_ritmo, 1 minuto di audio, {r.dispositivo} ===")
    print(f"caricamento modello {t_modello:.2f} s | prima analisi {t_primo:.2f} s")
    print(f"successive: media {statistics.mean(tempi):.2f} s (min {min(tempi):.2f}, max {max(tempi):.2f})"
          f" | BPM {r.bpm} {r.metro} {r.tonalita}")
    print(f"RAM aggiunta {(proc.memory_info().rss - ram0) / 2**20:.0f} MB | RAM processo "
          f"{proc.memory_info().rss / 2**20:.0f} MB")
    if dispositivo == "cuda":
        print(f"VRAM di picco {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


misura("cuda")
cuda_ok = torch.cuda.is_available
misura("cpu")
torch.cuda.is_available = cuda_ok

# --- Agente con i tool 1-4 ---
os.chdir(tmp)
from datapizza.agents import Agent  # noqa: E402
from datapizza.clients.openai import OpenAIClient  # noqa: E402

from app.tools.ritmo import analizza_ritmo_audio  # noqa: E402
from app.tools.separazione import separa_chitarra_audio  # noqa: E402
from app.tools.trascrizione import trascrivi_note_audio  # noqa: E402
from app.tools.validazione import valida_file_audio  # noqa: E402

ritmo._rilevatore = None
agente = Agent(
    name="analista_test",
    client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-nano"),
    system_prompt=(
        "Analizzi brani di chitarra. Procedura: 1) valida_file_audio; se non valido fermati. "
        "2) separa_chitarra_audio; se la chitarra non è presente fermati. "
        "3) trascrivi_note_audio sull''audio_consigliato'. "
        "4) analizza_ritmo_audio sul file ORIGINALE. Rispondi in una riga: "
        "'FATTO: <numero_note> note, <bpm> BPM, <metro>, tonalità <tonalita>/<tonalita_relativa>' "
        "oppure 'STOP: <motivo>'. Se ci sono avvisi sul tempo, aggiungi 'ATTENZIONE: tempo incerto'."
    ),
    tools=[valida_file_audio, separa_chitarra_audio, trascrivi_note_audio, analizza_ritmo_audio],
    max_steps=8,
)
print("\n=== Agente con tool 1-4 ===")
for f in (AUDIO / "mix.mp3", AUDIO / "solo_chitarra.mp3"):
    t0 = time.perf_counter()
    r = agente.run(f"Analizza il file: {f}")
    print(f">>> {f.name}: {r.text.strip()}")
    print(f"    {time.perf_counter() - t0:.2f}s | token in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
