"""Verifica preliminare dell'Orchestratore.

1. Modalità "codice" contro modalità "agente" sugli stessi file: stessa tablatura? token, tempo,
   avvisi raccolti.
2. Guardrail del resoconto LLM: 5 resoconti per brano, quanti scartati per numeri inventati.
3. Fermate e guasti: file finto, brano senza chitarra, errore iniettato in un passo, chiave assente.

Uso:  .venv\\Scripts\\python.exe scripts\\valuta_orchestratore.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.chdir(tempfile.mkdtemp())                     # la modalità agente scrive in ./output/<brano>
TMP = Path(tempfile.mkdtemp())
AUDIO = ROOT / "tests" / "audio"

import app.orchestratore as orch  # noqa: E402
from app.orchestratore import _dati_resoconto, _resoconto, numeri_inventati, trascrivi_brano  # noqa: E402


def eventi(file_json):
    tab = json.loads(Path(file_json).read_text(encoding="utf-8"))
    return [(b["numero"], ev["slot"], tuple((n["corda"], n["tasto"]) for n in ev["note"]))
            for b in tab["battute"] for ev in b["eventi"] if ev["note"]]


file_audio = [AUDIO / "mix.mp3"] + sorted((AUDIO / "reale").glob("*.mp3"))
print("=== 1. CODICE contro AGENTE ===")
for f in file_audio:
    c = trascrivi_brano(f, cartella_output=TMP / f"c_{f.stem}")
    a = trascrivi_brano(f, modalita="agente")
    uguali = c.completato and a.completato and eventi(c.file_tablatura) == eventi(a.file_tablatura)
    print(f"\n{f.name}")
    for r in (c, a):
        print(f"  {r.modalita:6s}: completato {r.completato}, {r.accordatura}, sens. {r.sensibilita}, "
              f"{r.numero_battute} battute, {r.numero_note} note | {r.tempo_totale_s} s | "
              f"token {r.token_in}+{r.token_out} = ${r.costo_usd:.4f} | {len(r.avvisi)} avvisi | span {len(r.passi)}")
    print(f"  stessa tablatura: {uguali} | avvisi solo in codice: {set(c.avvisi) - set(a.avvisi)} | "
          f"solo in agente: {set(a.avvisi) - set(c.avvisi)}")
    print(f"  resoconto codice: {c.resoconto}\n  resoconto agente: {a.resoconto}")

    print("  -- guardrail resoconto (5 prove) --")
    for k in range(5):
        testo, da_llm, originale_llm = _resoconto(c)
        if not da_llm:
            print(f"   prova {k + 1}: SCARTATO (numeri {numeri_inventati(originale_llm, _dati_resoconto(c))}): "
                  f"{originale_llm}")
        else:
            print(f"   prova {k + 1}: ok ({len(testo)} car.)")

print("\n=== 2. GUARDRAIL su testi costruiti ===")
dati = {"bpm": 107.9, "battute": 12, "note_in_tablatura": 136, "metro": "4/4"}
for t in ["Circa 108 BPM, 12 battute in 4/4 con 136 note.", "Tempo 107,9 BPM, due salti.",
          "Il brano ha 16 battute.", "Suona al 5° tasto il Do a 120 BPM."]:
    print(f"  {t!r}: inventati {numeri_inventati(t, dati) or 'nessuno'}")

print("\n=== 3. FERMATE E GUASTI ===")
finto = TMP / "finto.mp3"
finto.write_text("non sono un mp3")
r = trascrivi_brano(finto, cartella_output=TMP / "finto", resoconto_llm=False)
print(f"file finto: completato {r.completato}, stop: {r.motivo_stop} | passi {[p['nome'] for p in r.passi]}")

batteria = TMP / "solo_batteria.mp3"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(AUDIO / "gt_batteria.wav"), "-b:a", "192k",
                str(batteria)], check=True)
r = trascrivi_brano(batteria, cartella_output=TMP / "batteria", resoconto_llm=False)
print(f"solo batteria: completato {r.completato}, stop: {r.motivo_stop} | passi {[p['nome'] for p in r.passi]}")

originale = orch.crea_tablatura
orch.crea_tablatura = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("guasto simulato"))
r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=TMP / "guasto", resoconto_llm=False)
orch.crea_tablatura = originale
print(f"guasto nel Tablatore: completato {r.completato}, stop: {r.motivo_stop} | "
      f"span con errore: {[(p['nome'], p.get('errore')) for p in r.passi if p.get('esito') == 'errore']}")

chiave = os.environ.pop("OPENAI_API_KEY")
r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=TMP / "senza_chiave")
os.environ["OPENAI_API_KEY"] = chiave
print(f"senza chiave: completato {r.completato}, resoconto: {r.resoconto!r} | primo avviso: {r.avvisi[0]!r} | "
      f"costo ${r.costo_usd}")
print(f"\nesempio di dati passati all'LLM: {json.dumps(_dati_resoconto(c), ensure_ascii=False)[:400]}")
