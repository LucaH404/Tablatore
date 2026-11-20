"""Confronta più trascrizioni del brano intero di Nine Thou con LO STESSO riferimento
(allineato sulla chitarra della prima cartella), metriche A, B, C e metà brano.

Uso: .venv\\Scripts\\python.exe scripts\\confronta_banco.py <cartella_base> [altre cartelle...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from riferimento_nine_thou_completo import riferimento_temporale  # noqa: E402
from valuta_nine_thou_completo import valuta, valuta_tablatura  # noqa: E402

cartelle = [Path(a) for a in sys.argv[1:]]
rif, confini = riferimento_temporale(cartelle[0] / "chitarra.wav", cartelle[0] / "ritmo.json")
t_mezzo = confini[len(confini) // 2]
for d in cartelle:
    f = d / "note_power_chord.json" if (d / "note_power_chord.json").exists() else d / "note.json"
    note = json.loads(f.read_text(encoding="utf-8"))["note"]
    v = valuta(rif, note, confini)
    h = [valuta([n for n in rif if s(n["inizio"])], [n for n in note if s(n["inizio"])], confini)
         for s in (lambda t: t < t_mezzo, lambda t: t >= t_mezzo)]
    t = valuta_tablatura(rif, d / "tablatura_verificata.json", confini)
    print(f"\n### {d.name} ({f.name}, {t['accordatura']})")
    print(f"A richiamo {v['richiamo']:.1%} precisione {v['precisione']:.1%} | B esatto {v['attacco_esatto_richiamo']:.1%}"
          f" | C tab {t['richiamo']:.1%} (prec {t['precisione']:.1%})")
    print(f"  metà: 1a {h[0]['richiamo']:.1%}/{h[0]['precisione']:.1%}  2a {h[1]['richiamo']:.1%}/{h[1]['precisione']:.1%}")
    print(f"  perse {v['perse']}\n  in più {v['in_piu']}\n  sezioni {v['per_sezione']}")
