"""Esperimento Step 21: trascrivere separatamente le parti stereo della chitarra (sinistra, centro+destra...).

Per ogni parte (<lavoro>/stereo/chitarra_<parte>.wav) crea <lavoro>_<parte> con gli stessi passi 4-6
della pipeline: trascrizione (media, poi alta se poche note), power chord (auto), tablatura
(accordatura auto o indicata), verifica. Ritmo e basso sono quelli del lavoro.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_parti.py <lavoro> <accordatura|auto> <parte> [parte...]
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.tools.power_chord import completa_power_chord  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402


def main(lavoro: Path, accordatura: str, parti: list[str]):
    for parte in parti:
        out = lavoro.parent / f"{lavoro.name}_{parte}_{accordatura.replace(' ', '')}"
        out.mkdir(exist_ok=True)
        src = lavoro / "chitarra.wav" if parte == "intera" else lavoro / "stereo" / f"chitarra_{parte}.wav"
        shutil.copy(src, out / "chitarra.wav")
        for f in ("ritmo.json", "ritmo_tablatura.json"):
            if (lavoro / f).exists():
                shutil.copy(lavoro / f, out / f)
        rit = out / "ritmo_tablatura.json" if (out / "ritmo_tablatura.json").exists() else out / "ritmo.json"
        bpm = json.loads((lavoro / "ritmo.json").read_text(encoding="utf-8"))["bpm"]
        tr = trascrivi_note(out / "chitarra.wav", "media", out, bpm=bpm)
        if tr.note_per_beat is not None and tr.note_per_beat < config.NOTE_PER_BEAT_MIN:
            tr = trascrivi_note(out / "chitarra.wav", "alta", out, bpm=bpm)
        pc = completa_power_chord(out / "note.json", lavoro / "basso.wav", out / "chitarra.wav", "auto", out)
        tab = crea_tablatura(pc.file_note, rit, accordatura, out, titolo=parte)
        verifica_tablatura(tab.file_tablatura, cartella_output=out)
        print(f"{out.name}: sensibilità {tr.sensibilita}, note {tr.numero_note}, power chord "
              f"{pc.fonte or 'spenti'} ({pc.motivo[:60]}), accordatura {tab.accordatura}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
