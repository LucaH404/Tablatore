"""Prova della pagina web in un browser vero (Chrome headless): carica un MP3 sul server già
avviato, aspetta il risultato, poi salva uno screenshot della pagina e il PDF della vista di stampa.

Uso:  (server avviato con  python -m app.web.server)
      .venv\\Scripts\\python.exe scripts\\prova_pagina.py brano.mp3 cartella_uscita [accordatura] [separatore] [ripetizioni]
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
mp3, uscita = Path(sys.argv[1]), Path(sys.argv[2]).resolve()   # assoluto: Chrome non salva con percorsi relativi
accordatura = sys.argv[3] if len(sys.argv) > 3 else "auto"
separatore = sys.argv[4] if len(sys.argv) > 4 else "demucs"
ripetizioni = sys.argv[5] if len(sys.argv) > 5 else "false"
uscita.mkdir(parents=True, exist_ok=True)

with open(mp3, "rb") as f:
    r = httpx.post(f"{BASE}/api/lavori", files={"file": (mp3.name, f, "audio/mpeg")},
                   data={"accordatura": accordatura, "sensibilita": "auto", "resoconto_llm": "true",
                         "separatore": separatore, "ripetizioni": ripetizioni}, timeout=60)
r.raise_for_status()
id_lavoro = r.json()["id"]
t0, passi = time.time(), []
while True:
    s = httpx.get(f"{BASE}/api/lavori/{id_lavoro}").json()
    if s.get("passo") and (not passi or passi[-1] != s["passo"]):
        passi.append(s["passo"])
    if s["stato"] in ("finito", "errore"):
        break
    time.sleep(0.3)
print(f"lavoro {id_lavoro}: {s['stato']} in {time.time() - t0:.1f} s, passi visti {passi}")
ris = s.get("risultato", {})
print(f"  {ris.get('accordatura')} | {ris.get('bpm')} BPM | {ris.get('numero_battute')} battute | "
      f"${ris.get('costo_usd')} | resoconto: {ris.get('resoconto')}")

chrome = next(p for p in [shutil.which("chrome"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"] if p and Path(p).exists())
base_args = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={uscita / 'profilo'}",
             "--virtual-time-budget=40000", "--run-all-compositor-stages-before-draw", "--hide-scrollbars"]
subprocess.run(base_args + ["--window-size=1280,2200", f"--screenshot={uscita / 'pagina.png'}",
                            f"{BASE}/?lavoro={id_lavoro}"], capture_output=True)
subprocess.run(base_args + ["--no-pdf-header-footer", f"--print-to-pdf={uscita / 'tablatura.pdf'}",
                            f"{BASE}/stampa?lavoro={id_lavoro}&stampa=0"], capture_output=True)
for f in ("pagina.png", "tablatura.pdf"):
    p = uscita / f
    print(f"  {f}: {p.stat().st_size if p.exists() else 'MANCANTE'} byte")
print(id_lavoro)
