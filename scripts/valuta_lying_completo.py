"""Valutazione su tutto "Lying From You" (versione studio) con il riferimento Songsterr completo.

Metriche (stesse di Nine Thou, note stoppate escluse da A e C):
  A. note giuste nel beat giusto (richiamo / precisione);
  C. tablatura: stessa corda e tasto di Songsterr nel beat giusto;
  D. attacchi della tablatura abbinati uno a uno a quelli di Songsterr (entro 70 ms, stoppati compresi).
Il riferimento è allineato sulla chitarra della PRIMA cartella (griglia ritmo_tablatura.json se c'è).

Uso: .venv\\Scripts\\python.exe scripts\\valuta_lying_completo.py <cartella> [altre cartelle...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from metrica_attacchi import abbina, attacchi_tablatura  # noqa: E402
from riferimento_lying_completo import SEZIONI, riferimento_temporale  # noqa: E402
from valuta_nine_thou_completo import valuta, valuta_tablatura  # noqa: E402

FILE_NOTE = ("note_semplificate.json", "note_ripetizioni.json", "note_power_chord.json", "note.json")


def ritmo(d: Path) -> Path:
    return d / "ritmo_tablatura.json" if (d / "ritmo_tablatura.json").exists() else d / "ritmo.json"


def per_sezione_tab(rif, file_tab, confini):
    """Metrica C per sezione."""
    from valuta_nine_thou_completo import assegna_beat
    tab = json.loads(Path(file_tab).read_text(encoding="utf-8"))
    note = assegna_beat([dict(n, inizio=ev["tempo_s"]) for b in tab["battute"] for ev in b["eventi"]
                         for n in ev["note"]], confini)
    e_tab = defaultdict(set)
    for n in note:
        e_tab[n["beat"]].add((n["corda"], n["tasto"]))
    ok = defaultdict(lambda: [0, 0])
    for n in rif:
        if n["ghost"]:
            continue
        s = SEZIONI.get(n["disegno"], n["disegno"])
        ok[s][0] += (n["corda"], n["tasto"]) in e_tab.get(n["beat_globale"], set())
        ok[s][1] += 1
    return {s: f"{v[0] / v[1]:.0%} di {v[1]}" for s, v in ok.items()}


def main(cartelle):
    rif, confini = riferimento_temporale(cartelle[0] / "chitarra.wav", ritmo(cartelle[0]))
    rif_t = sorted({round(n["inizio"], 3) for n in rif})
    print(f"riferimento: {len(rif)} note ({sum(not n['ghost'] for n in rif)} con altezza), {len(rif_t)} attacchi, "
          f"battute audio {len(confini) // 4}; prima nota a {rif_t[0]:.2f} s")
    for d in cartelle:
        # il file di note usato è il più recente (le cartelle della CLI vengono riusate: restano file vecchi)
        f = max((d / x for x in FILE_NOTE if (d / x).exists()), key=lambda p: p.stat().st_mtime)
        v = valuta(rif, json.loads(f.read_text(encoding="utf-8"))["note"], confini)
        t = valuta_tablatura(rif, d / "tablatura_verificata.json", confini)
        our = attacchi_tablatura(d / "tablatura_verificata.json")
        ok = abbina(rif_t, our, 0.07)
        print(f"\n### {d.name} ({f.name}, {t['accordatura']})")
        print(f"A richiamo {v['richiamo']:.1%} precisione {v['precisione']:.1%} | C tab {t['richiamo']:.1%} "
              f"(prec {t['precisione']:.1%}) | D attacchi {len(our)} (rif {len(rif_t)}) richiamo {ok / len(rif_t):.1%} "
              f"precisione {ok / max(len(our), 1):.1%}")
        print(f"  perse {v['perse']}")
        print(f"  in più {v['in_piu']}")
        print(f"  sezioni (C) {per_sezione_tab(rif, d / 'tablatura_verificata.json', confini)}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
