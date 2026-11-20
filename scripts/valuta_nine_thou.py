"""Valuta la trascrizione sulle prime 3 battute dell'intro di "Nine thou" (riferimento da
tablatura Songsterr, vedi scripts/riferimento_nine_thou.py) con diverse impostazioni.

Metriche (attacco entro 50 ms):
  F1         : nota esatta
  F1 ottava  : nota giusta a meno dell'ottava (es. G#3 al posto di G#2)
  rich. G#2  : quante corde a vuoto (28 su 42) trovate, a meno dell'ottava
  rich. acute: quante note acute (14 su 42: G#3, F#3, A3) trovate esatte

Uso:  .venv\\Scripts\\python.exe scripts\\valuta_nine_thou.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from app.tools import trascrizione  # noqa: E402

CARTELLA = ROOT / "tests" / "audio" / "reale"
RIF = json.loads((CARTELLA / "gt_nine_thou_intro.json").read_text(encoding="utf-8"))
DURATA = 3 * 4 * 60 / RIF["bpm"]
modello = trascrizione._get_modello()
from basic_pitch.inference import run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402


def valuta(note_est, t0):
    gt = RIF["note"]
    # +1 s a tutti: mir_eval non accetta tempi negativi (note che anticipano di pochi ms)
    est = [(s - t0 + 1, e - t0 + 1, m) for s, e, m in note_est if -0.05 <= s - t0 < DURATA - 0.05]
    gi = np.array([[n["inizio"] + 1, n["fine"] + 1] for n in gt])
    gp = np.array([n["midi"] for n in gt])
    if not est:
        return 0, 0, 0, 0, 0
    ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    ep = np.array([m for *_, m in est])
    hz = mir_eval.util.midi_to_hz

    def f1(g_idx=None, ottava=False):
        gpp = (gp % 12 + 60) if ottava else gp
        epp = (ep % 12 + 60) if ottava else ep
        gi_, gpp_ = (gi, gpp) if g_idx is None else (gi[g_idx], gpp[g_idx])
        return mir_eval.transcription.precision_recall_f1_overlap(gi_, hz(gpp_), ei, hz(epp), offset_ratio=None)

    f = f1()[2]
    fo = f1(ottava=True)[2]
    aperte = np.where(gp == 44)[0]
    acute = np.where(gp != 44)[0]
    rich_aperte = f1(aperte, ottava=True)[1]
    rich_acute = f1(acute)[1]
    return f, fo, rich_aperte, rich_acute, len(est)


def trascrivi(uscita, onset, frame, ms, fmin, melodia):
    _, ev = model_output_to_notes(uscita, onset_thresh=onset, frame_thresh=frame,
                                  min_note_len=max(1, int(round(ms / 1000 * 22050 / 256))),
                                  min_freq=fmin, max_freq=1400, multiple_pitch_bends=False,
                                  melodia_trick=melodia)
    return [(s, e, m) for s, e, m, *_ in ev]


if __name__ == "__main__":
    combinazioni = [(0.8, 0.5, 128, 80, True)]                       # impostazioni attuali ("media")
    combinazioni += [(o, f, ms, 60, mel) for o in (0.6, 0.5, 0.4, 0.3) for f in (0.3, 0.2)
                     for ms in (128, 58) for mel in (True, False)]
    for nome in ("Nine thou - sintetizzato.mp3", "Nine-thou-reale.mp3"):
        t0 = RIF["primo_tempo_s"][nome]
        cartella = next(p for p in (ROOT / "output" / "reale").glob("*") if p.name.startswith(Path(nome).stem.replace(" ", "_")[:12]))
        chitarra = next(cartella.rglob("chitarra.wav"))
        uscita = run_inference(str(chitarra), modello)
        print(f"\n=== {nome} (traccia chitarra separata, primo tempo {t0} s) ===")
        print("onset frame ms  fmin melodia | note | F1   | F1 ottava | rich. G#2 | rich. acute")
        risultati = []
        for c in combinazioni:
            r = valuta(trascrivi(uscita, *c), t0)
            risultati.append((r, c))
        for r, c in [risultati[0]] + sorted(risultati[1:], key=lambda x: -x[0][1])[:8]:
            f, fo, ra, rac, n = r
            print(f"{c[0]:5} {c[1]:5} {c[2]:3} {c[3]:5} {str(c[4]):7} | {n:4d} | {f:.2f} | {fo:.2f}      | {ra:.2f}      | {rac:.2f}")
