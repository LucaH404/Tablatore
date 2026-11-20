"""Valutazione PER BATTUTA su "Dream On" (Lead Guitar di Songsterr), sezione per sezione.

Per ogni battuta audio (allineata alla partitura con il DTW di Nine Thou sulla chitarra della PRIMA
cartella) si confrontano:
  - note: insieme delle altezze MIDI (richiamo / precisione);
  - tasti: insieme (corda, tasto) della tablatura verificata (come la vede l'utente).
Uso: .venv\\Scripts\\python.exe scripts\\valuta_dream_on.py <cartella_rif> <cartella> [altre...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import riferimento_nine_thou_completo as base  # noqa: E402
from riferimento_dream_on import CORDE, SEZIONI_BATTUTE, battute_suonate, riferimento_battute  # noqa: E402


def ritmo(d: Path) -> Path:
    return d / "ritmo_tablatura.json" if (d / "ritmo_tablatura.json").exists() else d / "ritmo.json"


def main(rif_dir: Path, cartelle: list[Path]):
    bs = battute_suonate()
    mappa, limiti, _ = base.allinea(rif_dir / "chitarra.wav", ritmo(rif_dir), bs)
    rif = riferimento_battute()
    salti = sum(1 for a in range(1, len(mappa)) if mappa[a] != mappa[a - 1] + 1)
    print(f"battute audio {len(mappa)}, partitura 87; prima battuta audio -> {mappa[0] + 1}, "
          f"ultima -> {mappa[-1] + 1}; discontinuità nell'allineamento {salti}")
    for d in cartelle:
        tab = json.loads((d / "tablatura_verificata.json").read_text(encoding="utf-8"))
        eventi = [(e["tempo_s"], n) for b in tab["battute"] for e in b["eventi"] if "tempo_s" in e for n in e["note"]]
        per_sez = defaultdict(lambda: [0, 0, 0, 0, 0, 0])      # note: ok, rif, est | tasti: ok, rif, est
        for a, r in enumerate(mappa):
            b = r + 1
            vere = rif[b]
            nostre = [n for t, n in eventi if limiti[a] <= t < limiti[a + 1]]
            alt_v, alt_n = {CORDE[c] + f for c, f in vere}, {n["midi"] for n in nostre}
            tas_n = {(n["corda"], n["tasto"]) for n in nostre}
            s = per_sez[SEZIONI_BATTUTE[b]]
            s[0] += len(alt_v & alt_n); s[1] += len(alt_v); s[2] += len(alt_n)
            s[3] += len(vere & tas_n); s[4] += len(vere); s[5] += len(tas_n)
        tot = [sum(v[i] for v in per_sez.values()) for i in range(6)]
        pct = lambda x, y: f"{100 * x / y:3.0f}%" if y else "  — "
        print(f"\n### {d.name} ({tab['accordatura']})  note: richiamo {pct(tot[0], tot[1])} precisione "
              f"{pct(tot[0], tot[2])} | tasti: richiamo {pct(tot[3], tot[4])} precisione {pct(tot[3], tot[5])}")
        for nome in dict.fromkeys(SEZIONI_BATTUTE.values()):
            if nome in per_sez:
                v = per_sez[nome]
                print(f"   {nome:22s} note {pct(v[0], v[1])} / {pct(v[0], v[2])}   tasti {pct(v[3], v[4])} / {pct(v[3], v[5])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
