"""Taratura dei parametri di Basic Pitch sul brano sintetico.

Esegue il modello una sola volta per file, poi prova molte combinazioni di soglie
nella conversione in note (veloce) e misura precisione/richiamo/F1 sull'attacco delle note.
File valutati: solo_chitarra.mp3 e la chitarra separata da Demucs dal mix (come nella pipeline).
Uso:  .venv\\Scripts\\python.exe scripts\\taratura_trascrizione.py
"""
import itertools
import json
import logging
import sys
import tempfile
from pathlib import Path

import mir_eval
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import Model, run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402

from app.tools.separazione import separa_chitarra  # noqa: E402

AUDIO = ROOT / "tests" / "audio"
gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
ref_int = np.array([[n["inizio"], n["fine"]] for n in gt])
ref_hz = mir_eval.util.midi_to_hz(np.array([n["midi"] for n in gt]))

model = Model(ICASSP_2022_MODEL_PATH)
sep = separa_chitarra(AUDIO / "mix.mp3", tempfile.mkdtemp())
files = {"solo": str(AUDIO / "solo_chitarra.mp3"), "mix separato": sep.audio_chitarra}
uscite = {k: run_inference(v, model) for k, v in files.items()}


def valuta(out, onset, frame, min_ms, fmin, fmax):
    _, ev = model_output_to_notes(
        out, onset_thresh=onset, frame_thresh=frame,
        min_note_len=int(np.round(min_ms / 1000 * (22050 / 256))),
        min_freq=fmin, max_freq=fmax, multiple_pitch_bends=False, melodia_trick=True)
    if not ev:
        return 0, 0, 0, 0
    est_int = np.array([[e[0], e[1]] for e in ev])
    est_hz = mir_eval.util.midi_to_hz(np.array([e[2] for e in ev]))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_hz, est_int, est_hz, offset_ratio=None)
    return p, r, f, len(ev)


print("Default Basic Pitch (onset 0.5, frame 0.3, min 127.7 ms, nessun limite di frequenza):")
for k, out in uscite.items():
    p, r, f, n = valuta(out, 0.5, 0.3, 127.7, None, None)
    print(f"  {k:13} P={p:.2f} R={r:.2f} F1={f:.2f} note={n}")

risultati = []
for onset, frame, min_ms in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.3, 0.4, 0.5, 0.6], [58, 128]):
    punti = [valuta(out, onset, frame, min_ms, 80.0, 1400.0) for out in uscite.values()]
    risultati.append((np.mean([x[2] for x in punti]), onset, frame, min_ms, punti))
risultati.sort(reverse=True)

print("\nMigliori 6 combinazioni (range chitarra 80-1400 Hz), F1 medio sui due file:")
for f_med, onset, frame, min_ms, punti in risultati[:6]:
    dett = " | ".join(f"{k}: P={p:.2f} R={r:.2f} F1={f:.2f} n={n}" for k, (p, r, f, n) in zip(uscite, punti))
    print(f"  onset={onset} frame={frame} min={min_ms}ms -> F1 medio {f_med:.2f} || {dett}")
