"""Esperimento: power chord ricostruiti dalla chitarra stessa (senza basso) e tempo dimezzato.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_pc_chitarra.py <cartella_lavoro> <cartella_out> [dimezza]
Stampa la coerenza "interna" (gruppi di note che stanno in un power chord), la piattezza e le prime
battute della tablatura risultante.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.tools.power_chord import _gruppi, piattezza_spettrale  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402


def radice(gruppo, precedente=None):
    """Fondamentale (classe di altezza) del power chord compatibile con il gruppo, o None."""
    pcs = {n["midi"] % 12 for n in gruppo}
    basso = min(n["midi"] for n in gruppo) % 12
    candidati = [r for r in range(12) if pcs <= {r, (r + 7) % 12, (r + 4) % 12} and pcs & {r, (r + 7) % 12}]
    if not candidati:
        return None
    if len(pcs) == 1 and precedente is not None and (precedente + 7) % 12 in pcs:
        return precedente                              # quinta isolata subito dopo l'accordo: stesso accordo
    # preferisce la fondamentale presente, poi quella della nota più grave
    candidati.sort(key=lambda r: (r not in pcs, r != basso))
    return candidati[0]


def converti(note):
    gruppi = _gruppi(note)
    nuove, conv, multi, multi_ok, prec, t_prec = [], 0, 0, 0, None, -9
    for g in gruppi:
        t = g[0]["inizio"]
        p = prec if t - t_prec < 0.35 else None
        r = radice(g, p)
        if len({n["midi"] % 12 for n in g}) > 1:
            multi += 1
            multi_ok += r is not None
        if r is None:
            nuove += g
            prec = None
            continue
        m = config.RADICE_POWER_CHORD_MIN + (r - config.RADICE_POWER_CHORD_MIN) % 12
        base = min(g, key=lambda n: n["inizio"])
        fine = max(n["fine"] for n in g)
        nuove += [dict(base, midi=x, fine=fine, power_chord=True) for x in (m, m + 7, m + 12)]
        conv += 1
        prec, t_prec = r, t
    return nuove, conv, len(gruppi), multi_ok / max(multi, 1)


def main(lavoro, out, dimezza=False):
    out.mkdir(parents=True, exist_ok=True)
    dati = json.loads((lavoro / "note.json").read_text(encoding="utf-8"))
    nuove, conv, tot, coer = converti(dati["note"])
    print(f"piattezza {piattezza_spettrale(lavoro / 'chitarra.wav'):.5f}  coerenza interna {coer:.0%}  "
          f"convertiti {conv}/{tot}")
    (out / "note_power_chord.json").write_text(json.dumps(dict(dati, note=nuove)), encoding="utf-8")
    ritmo = json.loads((lavoro / "ritmo.json").read_text(encoding="utf-8"))
    if dimezza:
        primo = ritmo["beat"].index(ritmo["downbeat"][0])
        ritmo["beat"] = ritmo["beat"][primo % 2::2]
        ritmo["downbeat"] = ritmo["downbeat"][::2]
        ritmo["bpm"] = round(ritmo["bpm"] / 2, 1)
    (out / "ritmo.json").write_text(json.dumps(ritmo), encoding="utf-8")
    tab = crea_tablatura(out / "note_power_chord.json", out / "ritmo.json", "auto", out, titolo="prova")
    ver = verifica_tablatura(tab.file_tablatura, cartella_output=out)
    print("accordatura", tab.accordatura)
    righe = Path(ver.file_testo).read_text(encoding="utf-8").splitlines()
    print("\n".join(righe[:17]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), len(sys.argv) > 3)
