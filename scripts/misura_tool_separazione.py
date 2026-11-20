"""Misura qualità e consumi del Tool 2 (separa_chitarra) sul brano sintetico.

Qualità: SDR (Signal-to-Distortion Ratio, dB) tra chitarra separata e chitarra originale,
confrontato con il punto di partenza (usare il mix così com'è).
Consumi: tempo per minuto di audio, VRAM e RAM di picco, primo avvio vs avvii successivi.
Uso:  .venv\\Scripts\\python.exe scripts\\misura_tool_separazione.py
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from app.tools import separazione  # noqa: E402
from app.tools.audio_io import carica_audio, salva_wav  # noqa: E402

AUDIO = ROOT / "tests" / "audio"
tmp = Path(tempfile.mkdtemp())


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12))


def mono(path):
    return carica_audio(path, 44100, canali=1)[0]


gt = sf.read(AUDIO / "gt_chitarra.wav", dtype="float32")[0]

t0 = time.perf_counter()
separazione._get_separatore()
print(f"Caricamento modello: {time.perf_counter() - t0:.1f} s (solo al primo uso)")

# 1. Mix -> chitarra
torch.cuda.reset_peak_memory_stats()
r = separazione.separa_chitarra(AUDIO / "mix.mp3", tmp)
print("\n=== mix.mp3 (chitarra + basso + batteria) ===")
print(f"quota chitarra {r.quota_chitarra:.1%} | presente={r.chitarra_presente} | mix={r.era_un_mix}")
print("quote:", r.quote_energia)
print(f"SDR senza separazione (mix intero): {sdr(gt, mono(AUDIO / 'mix.mp3')):6.2f} dB")
print(f"SDR con separazione Demucs:          {sdr(gt, mono(r.audio_chitarra)):6.2f} dB")

# 2. Solo chitarra
r2 = separazione.separa_chitarra(AUDIO / "solo_chitarra.mp3", tmp)
print("\n=== solo_chitarra.mp3 ===")
print(f"quota chitarra {r2.quota_chitarra:.1%} | mix={r2.era_un_mix} | consigliato: {Path(r2.audio_consigliato).name}")

# 3. Nessuna chitarra
x = sf.read(AUDIO / "gt_basso.wav", dtype="float32")[0] + sf.read(AUDIO / "gt_batteria.wav", dtype="float32")[0]
no_chit = salva_wav(tmp / "senza_chitarra.wav", np.stack([x, x]), 44100)
r3 = separazione.separa_chitarra(no_chit, tmp)
print("\n=== basso + batteria (nessuna chitarra) ===")
print(f"quota chitarra {r3.quota_chitarra:.1%} | presente={r3.chitarra_presente} | avvisi={r3.avvisi}")

# 4. Consumi su 1 minuto di audio (mix ripetuto 3 volte)
uno_min = np.tile(carica_audio(AUDIO / "mix.mp3", 44100), 3)[:, : 60 * 44100]
f60 = salva_wav(tmp / "un_minuto.wav", uno_min, 44100)
proc = psutil.Process()
torch.cuda.reset_peak_memory_stats()
tempi = []
for _ in range(3):
    t0 = time.perf_counter()
    separazione.separa_chitarra(f60, tmp)
    tempi.append(time.perf_counter() - t0)
print("\n=== Consumi (1 minuto di audio, modello già caricato, 3 esecuzioni) ===")
print(f"tempo medio {np.mean(tempi):.2f} s (min {min(tempi):.2f}, max {max(tempi):.2f}) su {r.dispositivo}")
print(f"VRAM di picco {torch.cuda.max_memory_allocated() / 2**20:.0f} MB | RAM processo {proc.memory_info().rss / 2**20:.0f} MB")
