"""Esperimento Step 19: soglie di Basic Pitch (attacco / presenza / durata minima) contro gli attacchi doppi.

Il modello gira una volta per brano; per ogni combinazione si rifanno note, power chord (auto),
tablatura e verifica in <cartella_base>_s<onset>_<frame>_<durata>.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_soglie.py <cartella_base> [<cartella_base> ...]
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
from app.tools.trascrizione import _get_modello, nome_nota  # noqa: E402
from app.tools.verifica import verifica_tablatura  # noqa: E402

COMBINAZIONI = [(0.5, 0.3, 58), (0.6, 0.3, 58), (0.7, 0.3, 58), (0.8, 0.3, 58), (0.6, 0.3, 100), (0.7, 0.3, 100),
                (0.8, 0.5, 128)]
DIMEZZA = "--dimezza" in sys.argv          # tempo dimezzato (registrazioni senza batteria a tempo doppio)


def dimezza(ritmo):
    primo = ritmo["beat"].index(ritmo["downbeat"][0])
    return dict(ritmo, beat=ritmo["beat"][primo % 2::2], downbeat=ritmo["downbeat"][::2],
                bpm=round(ritmo["bpm"] / 2, 1))


def main(basi):
    modello = _get_modello()
    from basic_pitch.inference import run_inference
    from basic_pitch.note_creation import model_output_to_notes
    for base in basi:
        uscita = run_inference(str(base / "chitarra.wav"), modello)
        for onset, frame, dur in COMBINAZIONI:
            out = base.parent / f"{base.name}{'_mezzo' if DIMEZZA else ''}_s{onset}_{frame}_{dur}"
            out.mkdir(exist_ok=True)
            ritmo = json.loads((base / "ritmo.json").read_text(encoding="utf-8"))
            (out / "ritmo.json").write_text(json.dumps(dimezza(ritmo) if DIMEZZA else ritmo), encoding="utf-8")
            _, eventi = model_output_to_notes(
                uscita, onset_thresh=onset, frame_thresh=frame,
                min_note_len=int(round(dur / 1000 * config.FPS_BASIC_PITCH)),
                min_freq=config.FREQ_MIN_HZ, max_freq=config.FREQ_MAX_HZ,
                multiple_pitch_bends=False, melodia_trick=True)
            note = sorted(({"inizio": round(float(s), 3), "fine": round(float(e), 3), "midi": int(m),
                            "nome": nome_nota(int(m)), "intensita": round(float(a), 3)} for s, e, m, a, _ in eventi),
                          key=lambda n: (n["inizio"], n["midi"]))
            (out / "note.json").write_text(json.dumps({"sorgente": "chitarra.wav", "note": note}), encoding="utf-8")
            basso = base / "basso.wav"
            pc = completa_power_chord(out / "note.json", basso if basso.exists() else None, base / "chitarra.wav",
                                      "auto", out)
            if not pc.attiva:          # per il confronto serve sempre note_power_chord.json
                shutil.copy(out / "note.json", out / "note_power_chord.json")
            tab = crea_tablatura(out / "note_power_chord.json", out / "ritmo.json", "auto", out, titolo="soglie")
            verifica_tablatura(tab.file_tablatura, cartella_output=out)
            print(f"{out.name}: note {len(note)}, power chord {pc.fonte or 'spenti'} ({pc.attacchi_convertiti}/"
                  f"{pc.attacchi_totali}), accordatura {tab.accordatura}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:] if not a.startswith("--")])
