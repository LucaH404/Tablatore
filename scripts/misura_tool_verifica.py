"""Misura i consumi del Tool 6 (verifica_tablatura) e prova un agente con i tool 1-6.

Uso:  .venv\\Scripts\\python.exe scripts\\misura_tool_verifica.py
"""
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
tmp = Path(tempfile.mkdtemp())
AUDIO = ROOT / "tests" / "audio"
proc = psutil.Process()

from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402

# stesso caso pesante del Tool 5: 1 minuto, 384 note (accordi da 2), più 20 armoniche spurie
gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
periodo = 8 * 4 * 0.6
note = [dict(n, inizio=n["inizio"] + k * periodo, fine=n["fine"] + k * periodo, midi=n["midi"] + d, intensita=0.8)
        for k in range(3) for n in gt for d in (0, 12)]
note += [dict(note[i * 19], midi=88, intensita=0.4) for i in range(20)]
(tmp / "note.json").write_text(json.dumps({"sorgente": "un_minuto", "note": note}))
beat = [round(i * 0.6, 3) for i in range(101)]
(tmp / "ritmo.json").write_text(json.dumps({"bpm": 100, "metro": "4/4", "beat": beat, "downbeat": beat[::4]}))
r = crea_tablatura(tmp / "note.json", tmp / "ritmo.json", "standard", tmp)

ram0 = proc.memory_info().rss
t = []
for _ in range(5):
    t0 = time.perf_counter()
    v = verifica_tablatura(r.file_tablatura)
    t.append(time.perf_counter() - t0)
print(f"verifica_tablatura: media {statistics.mean(t):.3f} s (min {min(t):.3f}, max {max(t):.3f}) | "
      f"{r.numero_note} note -> tolte {v.note_tolte}, problemi {v.problemi_iniziali} -> {v.problemi_finali}, "
      f"{v.cicli} cicli | RAM aggiunta {(proc.memory_info().rss - ram0) / 2**20:.0f} MB")

# --- Agente con i tool 1-6 ---
os.chdir(tmp)
from datapizza.agents import Agent  # noqa: E402
from datapizza.clients.openai import OpenAIClient  # noqa: E402

from app.tools.ritmo import analizza_ritmo_audio  # noqa: E402
from app.tools.separazione import separa_chitarra_audio  # noqa: E402
from app.tools.tablatura import crea_tablatura_audio  # noqa: E402
from app.tools.trascrizione import trascrivi_note_audio  # noqa: E402
from app.tools.validazione import valida_file_audio  # noqa: E402
from app.tools.verifica import verifica_tablatura_audio  # noqa: E402

agente = Agent(
    name="tablatore_test",
    client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-nano"),
    system_prompt=(
        "Crei tablature di chitarra da file audio. Procedura: 1) valida_file_audio; se non valido fermati. "
        "2) separa_chitarra_audio; se la chitarra non è presente fermati. "
        "3) analizza_ritmo_audio sul file ORIGINALE. "
        "4) trascrivi_note_audio sull''audio_consigliato' passando il 'bpm'; se gli avvisi suggeriscono "
        "un'altra sensibilità riprova una sola volta. "
        "5) crea_tablatura_audio con 'file_note' e 'file_ritmo'. "
        "6) verifica_tablatura_audio con il 'file_tablatura'. Rispondi in una riga: "
        "'FATTO: <numero_battute> battute, accordatura <accordatura>, sensibilità <s>, suonabile <sì/no>, "
        "note tolte <n>, file <file_gp5 verificato>' oppure 'STOP: <motivo>'."
    ),
    tools=[valida_file_audio, separa_chitarra_audio, analizza_ritmo_audio, trascrivi_note_audio,
           crea_tablatura_audio, verifica_tablatura_audio],
    max_steps=12,
)
print("\n=== Agente con tool 1-6 ===")
for f in [AUDIO / "mix.mp3"] + sorted((AUDIO / "reale").glob("*.mp3")):
    t0 = time.perf_counter()
    risp = agente.run(f"Crea la tablatura del file: {f}")
    print(f">>> {f.name}: {risp.text.strip()}")
    print(f"    {time.perf_counter() - t0:.2f}s | token in={risp.usage.prompt_tokens} out={risp.usage.completion_tokens}")
