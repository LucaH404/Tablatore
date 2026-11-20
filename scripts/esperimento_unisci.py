"""Esperimento Step 19: unire all'accordo precedente le "continuazioni" (stesso accordo ancora in suono,
attaccato a meno di <soglia> beat) e rifare tablatura + verifica.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_unisci.py <cartella_base> <cartella_out> <soglia_beat>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.tools.power_chord import _gruppi  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402


def unisci(note, beat, soglia):
    gruppi = _gruppi(note)
    tenuti, uniti = [gruppi[0]], 0
    for g in gruppi[1:]:
        p = tenuti[-1]
        t0, t1 = p[0]["inizio"], g[0]["inizio"]
        k = int(np.clip(np.searchsorted(beat, t1) - 1, 0, len(beat) - 2))
        dist = (t1 - t0) / (beat[k + 1] - beat[k])
        stessa = {n["midi"] for n in g} == {n["midi"] for n in p}
        suona = max(n["fine"] for n in p) >= t1 - 0.03
        if stessa and suona and dist < soglia:
            fine = max(n["fine"] for n in g)
            for n in p:
                n["fine"] = max(n["fine"], fine)
            uniti += 1
        else:
            tenuti.append(g)
    return [n for g in tenuti for n in g], uniti, len(gruppi)


def main(base, out, soglia):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(base / "ritmo.json", out / "ritmo.json")
    dati = json.loads((base / "note_power_chord.json").read_text(encoding="utf-8"))
    beat = np.array(json.loads((base / "ritmo.json").read_text(encoding="utf-8"))["beat"])
    nuove, uniti, tot = unisci(dati["note"], beat, soglia)
    (out / "note_power_chord.json").write_text(json.dumps(dict(dati, note=nuove)), encoding="utf-8")
    tab = crea_tablatura(out / "note_power_chord.json", out / "ritmo.json", "auto", out, titolo="unisci")
    verifica_tablatura(tab.file_tablatura, cartella_output=out)
    print(f"{out.name}: uniti {uniti} accordi su {tot} -> {tot - uniti}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))
