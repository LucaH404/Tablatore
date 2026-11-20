"""Step 24: effetto del passo 4c (confronto tra ripetizioni) su una cartella già trascritta.

Copia la cartella in <cartella>_rip, applica vota_ripetizioni alle note usate dalla tablatura
(note_ottave.json se c'è, altrimenti note_power_chord.json o note.json), poi tablatura e verifica.
Uso: .venv\\Scripts\\python.exe scripts\\esperimento_ripetizioni_parte.py <cartella> <basso.wav> <accordatura|auto>
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.tools.ripetizioni import vota_ripetizioni  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402


def main(src: Path, basso: Path, accordatura: str):
    out = src.parent / f"{src.name}_rip"
    out.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".json", ".wav"):
            shutil.copy(f, out / f.name)
    note = next(out / n for n in ("note_ottave.json", "note_power_chord.json", "note.json") if (out / n).exists())
    rit = out / "ritmo_tablatura.json" if (out / "ritmo_tablatura.json").exists() else out / "ritmo.json"
    rp = vota_ripetizioni(note, rit, out / "chitarra.wav", basso, out)
    tab = crea_tablatura(rp.file_note, rit, accordatura, out)
    verifica_tablatura(tab.file_tablatura, cartella_output=out)
    print(f"{out.name}: da {note.name}; battute con ripetizioni {rp.battute_con_ripetizioni}/{rp.battute}, "
          f"+{rp.note_aggiunte} -{rp.note_tolte} note, accordatura {tab.accordatura}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
