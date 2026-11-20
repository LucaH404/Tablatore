"""Esperimento Step 22: correggere le note lette UN'OTTAVA SOPRA (O73, Dream On).

Nella parte stereo separata il F4 dell'arpeggio esce spesso come F5: la maschera conserva meglio la
seconda armonica della fondamentale. La nota "su" è più debole delle vicine e la sua ottava sotto è
suonata poco prima/dopo. Regola provata: una nota si abbassa di 12 se
  - nella finestra di ±W secondi l'ottava sotto compare almeno K volte,
  - la sua intensità è sotto la mediana della finestra di almeno D,
  - (opzionale) è sopra l'80° percentile delle altezze della finestra.
Poi passi 4b-6 della pipeline come in esperimento_parti.py.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_ottava_su.py <cartella_parte> <lavoro> <W> <K> <D> [suffisso]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.tools.power_chord import completa_power_chord  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.trascrizione import nome_nota  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402


SOLA_S = float(__import__("os").environ.get("SOLA_S", "0.06"))   # nota "da sola": nessun altro attacco così vicino


def correggi(note: list[dict], w: float, k: int, d: float, percentile: bool = True) -> tuple[list[dict], int]:
    nuove, corrette = [], 0
    for n in note:
        vicine = [v for v in note if abs(v["inizio"] - n["inizio"]) <= w and v is not n]
        sotto = sum(1 for v in vicine if v["midi"] == n["midi"] - 12)
        insieme = any(abs(v["inizio"] - n["inizio"]) < SOLA_S for v in vicine)   # parte di un accordo
        if insieme:
            sotto = 0
        med = np.median([v["intensita"] for v in vicine]) if vicine else 1.0
        alto = not percentile or not vicine or n["midi"] >= np.percentile([v["midi"] for v in vicine], 80)
        m = dict(n)
        if sotto >= k and n["intensita"] < med - d and alto:
            m["midi"] -= 12
            m["nome"] = nome_nota(m["midi"])
            corrette += 1
        nuove.append(m)
    return nuove, corrette


def main(parte: Path, lavoro: Path, w: float, k: int, d: float, suff: str):
    out = parte.parent / f"{parte.name}_{suff}"
    out.mkdir(exist_ok=True)
    for f in ("chitarra.wav", "ritmo.json", "ritmo_tablatura.json"):
        if (parte / f).exists():
            shutil.copy(parte / f, out / f)
    dati = json.loads((parte / "note.json").read_text(encoding="utf-8"))
    note, corrette = correggi(dati["note"], w, k, d)
    dati["note"] = note
    (out / "note.json").write_text(json.dumps(dati, ensure_ascii=False), encoding="utf-8")
    rit = out / "ritmo_tablatura.json" if (out / "ritmo_tablatura.json").exists() else out / "ritmo.json"
    pc = completa_power_chord(out / "note.json", lavoro / "basso.wav", out / "chitarra.wav", "auto", out)
    acc = json.loads((parte / "tablatura_verificata.json").read_text(encoding="utf-8"))["accordatura"]
    tab = crea_tablatura(pc.file_note, rit, "standard" if acc.startswith("standard") or "standard" in parte.name else "auto", out)
    verifica_tablatura(tab.file_tablatura, cartella_output=out)
    print(f"{out.name}: corrette {corrette}/{len(note)} note, accordatura {tab.accordatura}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]),
         sys.argv[6] if len(sys.argv) > 6 else "ott")
