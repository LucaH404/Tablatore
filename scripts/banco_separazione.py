"""Banco di prova: stessi passi 4-6 della pipeline con chitarra e basso presi da un altro separatore.

Uso: .venv\\Scripts\\python.exe scripts\\banco_separazione.py <cartella_base> <chitarra.wav> <basso.wav> <cartella_nuova>
La cartella base è un lavoro completo (ritmo.json, originale.wav); nella nuova si rifanno
trascrizione (sensibilità media/auto), power chord (auto), tablatura (accordatura auto), verifica.
"""
import shutil
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.tools.power_chord import completa_power_chord  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402
import json  # noqa: E402


def rifai(base: Path, chitarra: Path, basso: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in ("ritmo.json", "originale.wav"):
        shutil.copy(base / f, out / f)
    for src, dst in ((chitarra, "chitarra.wav"), (basso, "basso.wav")):
        y, sr = sf.read(src)
        sf.write(out / dst, y, sr, subtype="PCM_16")
    for f in ("linea_basso.npz",):
        (out / f).unlink(missing_ok=True)
    bpm = json.loads((out / "ritmo.json").read_text(encoding="utf-8"))["bpm"]
    tr = trascrivi_note(out / "chitarra.wav", "media", out, bpm=bpm)
    if tr.note_per_beat is not None and tr.note_per_beat < config.NOTE_PER_BEAT_MIN:
        tr = trascrivi_note(out / "chitarra.wav", "alta", out, bpm=bpm)
    pc = completa_power_chord(out / "note.json", out / "basso.wav", out / "chitarra.wav", "auto", out)
    tab = crea_tablatura(pc.file_note, out / "ritmo.json", "auto", out, titolo="banco")
    ver = verifica_tablatura(tab.file_tablatura, cartella_output=out)
    print(f"sensibilità {tr.sensibilita}, note {tr.numero_note}, power chord {pc.attiva} ({pc.motivo}), "
          f"accordatura {tab.accordatura}, note in tablatura {ver.numero_note}")


if __name__ == "__main__":
    rifai(*(Path(a) for a in sys.argv[1:5]))
