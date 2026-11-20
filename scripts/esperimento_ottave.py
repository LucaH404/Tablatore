"""Esperimento Step 20 (O70): non trasformare in power chord le OTTAVE (stessa nota a 12 semitoni,
senza quinta, nota più grave >= soglia). Rifà power chord (auto), tablatura e verifica in
<cartella>_ott<soglia> per ogni soglia; "off" = regola spenta (come ora).

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_ottave.py <cartella> [altre...]
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.tools import power_chord as P  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402

SOGLIE = ["off", 48, 50, 52]


def main(cartelle):
    for d in cartelle:
        for s in SOGLIE:
            config.OTTAVA_MIDI_MIN = None if s == "off" else s
            out = d.parent / f"{d.name}_ott{s}"
            out.mkdir(exist_ok=True)
            for f in ("ritmo.json", "ritmo_tablatura.json", "chitarra.wav"):
                if (d / f).exists() and f != "chitarra.wav":
                    shutil.copy(d / f, out / f)
            basso = d / "basso.wav"
            pc = P.completa_power_chord(d / "note.json", basso if basso.exists() else None, d / "chitarra.wav",
                                        "auto", out)
            if not pc.attiva:
                shutil.copy(d / "note.json", out / "note_power_chord.json")
            rit = out / "ritmo_tablatura.json" if (out / "ritmo_tablatura.json").exists() else out / "ritmo.json"
            tab = crea_tablatura(out / "note_power_chord.json", rit, "auto", out, titolo="ottave")
            verifica_tablatura(tab.file_tablatura, cartella_output=out)
            print(f"{out.name}: power chord {pc.fonte} {pc.attacchi_convertiti}/{pc.attacchi_totali}, "
                  f"accordatura {tab.accordatura}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
