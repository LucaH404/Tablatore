"""Genera un brano di prova sintetico con tracce separate note (ground truth).

Produce in tests/audio/:
  - gt_chitarra.wav  : chitarra pizzicata (Karplus-Strong), arpeggi Am - F - C - G
  - gt_basso.wav     : basso sulle toniche
  - gt_batteria.wav  : cassa, rullante, charleston
  - mix.mp3          : somma delle tre tracce (quello che caricherebbe l'utente)
  - solo_chitarra.mp3: solo la chitarra (caso "nessun mix")
  - gt_note.json     : le note suonate dalla chitarra (MIDI, inizio, fine) = partitura di riferimento
Servono a misurare quanto bene Demucs isola la chitarra (confronto con gt_chitarra).
Uso:  .venv\\Scripts\\python.exe scripts\\genera_mix_test.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 100
BEAT = 60 / BPM
OUT = Path(__file__).resolve().parents[1] / "tests" / "audio"
rng = np.random.default_rng(42)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def karplus_strong(freq, dur, decay=0.996):
    """Corda pizzicata: rumore filtrato in un ritardo pari al periodo della nota."""
    n = int(SR * dur)
    period = max(2, int(SR / freq))
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = decay * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return out


def piazza(traccia, suono, t):
    i = int(t * SR)
    fine = min(len(traccia), i + len(suono))
    traccia[i:fine] += suono[: fine - i]


battute = 8
durata = battute * 4 * BEAT + 1.0
n = int(SR * durata)
chitarra, basso, batteria = np.zeros(n), np.zeros(n), np.zeros(n)
note_gt = []  # partitura di riferimento della chitarra

# Accordi (note MIDI di un arpeggio a 4 note) e toniche del basso
accordi = [([57, 64, 69, 72], 45), ([53, 60, 65, 69], 41), ([48, 55, 60, 64], 36), ([55, 62, 67, 71], 43)]
for b in range(battute):
    note, tonica = accordi[b % 4]
    t0 = b * 4 * BEAT
    for k in range(8):  # arpeggio a crome
        m = note[[0, 1, 2, 3, 2, 1, 2, 3][k]]
        piazza(chitarra, 0.5 * karplus_strong(midi_hz(m), 1.2), t0 + k * BEAT / 2)
        note_gt.append({"midi": m, "inizio": round(t0 + k * BEAT / 2, 4),
                        "fine": round(t0 + (k + 1) * BEAT / 2, 4)})
    for k in range(4):  # basso: semiminime, sinusoide con armoniche e inviluppo
        t = np.arange(int(SR * BEAT * 0.9)) / SR
        f = midi_hz(tonica)
        s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-3 * t)
        piazza(basso, 0.5 * s, t0 + k * BEAT)
    for k in range(4):  # batteria
        t = np.arange(int(SR * 0.25)) / SR
        if k in (0, 2):  # cassa: sinusoide che scende di frequenza
            piazza(batteria, 0.9 * np.sin(2 * np.pi * (50 + 100 * np.exp(-30 * t)) * t) * np.exp(-12 * t), t0 + k * BEAT)
        else:  # rullante: rumore
            piazza(batteria, 0.4 * rng.uniform(-1, 1, len(t)) * np.exp(-20 * t), t0 + k * BEAT)
        for h in range(2):  # charleston: rumore breve
            th = np.arange(int(SR * 0.05)) / SR
            piazza(batteria, 0.12 * rng.uniform(-1, 1, len(th)) * np.exp(-60 * th), t0 + k * BEAT + h * BEAT / 2)

mix = chitarra + basso + batteria
picco = np.abs(mix).max() / 0.9
OUT.mkdir(parents=True, exist_ok=True)
for nome, x in [("gt_chitarra", chitarra), ("gt_basso", basso), ("gt_batteria", batteria), ("mix", mix)]:
    sf.write(OUT / f"{nome}.wav", (x / picco).astype(np.float32), SR)

ff = shutil.which("ffmpeg")
for src, dst in [("mix.wav", "mix.mp3"), ("gt_chitarra.wav", "solo_chitarra.mp3")]:
    subprocess.run([ff, "-y", "-v", "error", "-i", str(OUT / src), "-b:a", "192k", str(OUT / dst)], check=True)
(OUT / "mix.wav").unlink()
(OUT / "gt_note.json").write_text(json.dumps({"bpm": BPM, "tonalita": "A minor", "note": note_gt}, indent=1))
print(f"Generati in {OUT}: durata {durata:.1f} s, {BPM} BPM, accordi Am-F-C-G")
