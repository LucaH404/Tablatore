"""Classifica le note 'in più' trovate da Basic Pitch rispetto alla partitura di riferimento.

Uso:  .venv\\Scripts\\python.exe scripts\\analizza_falsi_trascrizione.py [file] [onset] [frame]
"""
import collections
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import predict  # noqa: E402

AUDIO = Path(__file__).resolve().parents[1] / "tests" / "audio"
f = sys.argv[1] if len(sys.argv) > 1 else "solo_chitarra.mp3"
gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
_, _, ev = predict(str(AUDIO / f), ICASSP_2022_MODEL_PATH)


def vera(s, p):
    return any(abs(n["inizio"] - s) < 0.05 and n["midi"] == p for n in gt)


c = collections.Counter()
falsi, veri = [], []
for s, e, p, a, _ in ev:
    if vera(s, p):
        veri.append((e - s, a))
        continue
    falsi.append((e - s, a))
    # note che stanno ancora suonando in quell'istante (la corda pizzicata risuona ~1,2 s)
    risonanti = [n for n in gt if n["inizio"] - 0.05 <= s <= n["inizio"] + 1.2]
    armoniche = {12, 19, 24, 28, 31, 34, 36}  # 2a, 3a, 4a, 5a, 6a, 7a, 8a armonica
    if any(n["midi"] == p for n in risonanti):
        c["nota vera che risuona, ri-attaccata (doppione)"] += 1
    elif any(p - n["midi"] in armoniche for n in risonanti):
        c["armonica di una nota che suona (ottava, quinta...)"] += 1
    elif any(n["midi"] - p in (12, 24) for n in risonanti):
        c["ottava sotto una nota che suona"] += 1
    else:
        c["altro"] += 1

print(f"{f}: {len(veri)} vere, {len(falsi)} false")
for k, v in c.most_common():
    print(f"{v:4}  {k}")
fv, ff = np.array(veri), np.array(falsi)
print(f"note vere : durata media {fv[:, 0].mean():.2f}s, ampiezza media {fv[:, 1].mean():.2f}")
print(f"note false: durata media {ff[:, 0].mean():.2f}s, ampiezza media {ff[:, 1].mean():.2f}")
