"""Verifica che un agente datapizza-ai usi correttamente i tool 1 e 2 in sequenza.

Casi: mix con chitarra (deve validare e poi separare), brano senza chitarra
(deve validare, separare e fermarsi segnalando l'assenza), file finto (deve fermarsi
alla validazione senza chiamare la separazione).
Uso:  .venv\\Scripts\\python.exe scripts\\prova_agente_tool12.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(tempfile.mkdtemp())  # gli output dei tool finiscono in una cartella temporanea

from datapizza.agents import Agent  # noqa: E402
from datapizza.clients.openai import OpenAIClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.tools.separazione import separa_chitarra_audio  # noqa: E402
from app.tools.validazione import valida_file_audio  # noqa: E402

load_dotenv(ROOT / ".env")
AUDIO = ROOT / "tests" / "audio"
tmp = Path.cwd()

senza = tmp / "senza_chitarra.mp3"
subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-i", str(AUDIO / "gt_basso.wav"),
                "-i", str(AUDIO / "gt_batteria.wav"), "-filter_complex", "amix=inputs=2",
                "-b:a", "192k", str(senza)], check=True)
finto = tmp / "finto.mp3"
finto.write_text("non sono audio " * 50)

agente = Agent(
    name="analista_test",
    client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-nano"),
    system_prompt=(
        "Prepari i brani per la trascrizione di chitarra. Procedura: 1) valida il file con "
        "valida_file_audio; se non è valido fermati. 2) isola la chitarra con separa_chitarra_audio; "
        "se la chitarra non è presente fermati. Rispondi in una riga: 'PRONTO: <audio_consigliato>' "
        "oppure 'STOP: <motivo>'."
    ),
    tools=[valida_file_audio, separa_chitarra_audio],
    max_steps=5,
)

for f in (AUDIO / "mix.mp3", senza, finto):
    t0 = time.perf_counter()
    r = agente.run(f"Prepara il file: {f}")
    print(f"\n>>> {f.name}: {r.text.strip()}")
    print(f"    {time.perf_counter() - t0:.2f}s | token in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
