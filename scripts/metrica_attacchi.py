"""Metrica D (Step 19): attacchi abbinati uno a uno con il riferimento Songsterr di Nine Thou.

Le metriche A e C confrontano l'insieme di note in ogni beat: un accordo ripetuto tre volte nello
stesso beat conta come uno solo, quindi le note in più "per riempire il tempo" non si vedevano.
Qui ogni attacco del riferimento si abbina al massimo a un attacco trascritto (entro 50 ms):
i doppioni restano senza abbinamento e abbassano la precisione.

Uso: .venv\\Scripts\\python.exe scripts\\metrica_attacchi.py <cartella_base> [altre cartelle...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from app.tools.power_chord import _gruppi  # noqa: E402
from riferimento_nine_thou_completo import riferimento_temporale  # noqa: E402

TOLLERANZA = 0.05


def abbina(rif_t, our_t, tol=TOLLERANZA):
    """Abbinamento uno a uno (greedy per distanza crescente)."""
    coppie = sorted((abs(a - b), i, j) for i, a in enumerate(rif_t) for j, b in enumerate(our_t)
                    if abs(a - b) <= tol)
    usati_r, usati_o = set(), set()
    for _, i, j in coppie:
        if i not in usati_r and j not in usati_o:
            usati_r.add(i)
            usati_o.add(j)
    return len(usati_r)


def attacchi_tablatura(f):
    """Tempi degli attacchi scritti nella tablatura (dopo quantizzazione e verifica): quello che vede
    l'utente. Il tempo è quello della semicroma della griglia, quindi l'abbinamento ha più margine."""
    tab = json.loads(Path(f).read_text(encoding="utf-8"))
    return sorted(e["tempo_s"] for b in tab["battute"] for e in b["eventi"] if e["note"] and "tempo_s" in e)


def riga(nome, rif_t, our_t, tol):
    ok = abbina(rif_t, our_t, tol)
    print(f"{nome:34s} attacchi {len(our_t):5d} (rif {len(rif_t)}) | richiamo {ok / len(rif_t):.1%} "
          f"precisione {ok / len(our_t):.1%} | F1 {2 * ok / (len(rif_t) + len(our_t)):.1%}")


def main(cartelle):
    rif, _ = riferimento_temporale(cartelle[0] / "chitarra.wav", cartelle[0] / "ritmo.json")
    rif_t = sorted({round(n["inizio"], 3) for n in rif})
    for d in cartelle:
        f = d / "note_power_chord.json"
        if f.exists():
            note = json.loads(f.read_text(encoding="utf-8"))["note"]
            riga(d.name + " (note)", rif_t, [g[0]["inizio"] for g in _gruppi(note)], TOLLERANZA)
        riga(d.name + " (tablatura)", rif_t, attacchi_tablatura(d / "tablatura_verificata.json"), 0.07)


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
