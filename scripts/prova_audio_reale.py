"""Prova i tool 1-4 sui file in tests/audio/reale/ e stampa un riepilogo.

Uso:  .venv\\Scripts\\python.exe scripts\\prova_audio_reale.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.ritmo import analizza_ritmo  # noqa: E402
from app.tools.separazione import separa_chitarra  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402
from app.tools.validazione import valida_file  # noqa: E402

OUT = ROOT / "output" / "reale"
for f in sorted((ROOT / "tests" / "audio" / "reale").glob("*.mp3")):
    print(f"\n######## {f.name}")
    v = valida_file(f)
    print("VALIDAZIONE:", v.valido, v.errori, v.avvisi[:-2] if len(v.avvisi) > 2 else [], v.metadati)
    if not v.valido:
        continue
    cartella = OUT / f.stem.replace(" ", "_")
    s = separa_chitarra(f, cartella)
    print("SEPARAZIONE:", {k: v for k, v in asdict(s).items() if k not in ("audio_chitarra",)})
    for sens in ("bassa", "media", "alta"):
        t = trascrivi_note(s.audio_consigliato, sens, cartella / f"note_{sens}")
        print(f"TRASCRIZIONE {sens}:", {k: v for k, v in asdict(t).items() if k not in ("file_note", "file_midi")})
    r = analizza_ritmo(f, cartella)
    print("RITMO:", {k: v for k, v in asdict(r).items() if k != "file_ritmo"})
