"""Step 24: riconoscere "due chitarre diverse" (Dream On) da "stessa parte registrata due volte" (Lying, Nine Thou).

Per finestre di W secondi confronta il cromagramma (quali note suonano) del canale sinistro e destro
della chitarra isolata: similitudine coseno. Parte raddoppiata → simile quasi ovunque; due parti
diverse → tratti con similitudine bassa. Stampa la distribuzione e la quota di finestre "diverse".

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_due_chitarre.py <cartella> [altre...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

W = 2.0          # secondi per finestra
SOGLIA = 0.8     # sotto: sinistra e destra suonano note diverse


def similitudini(wav_a: Path, wav_b: Path | None = None) -> np.ndarray:
    """Due file (parti separate, sommate a mono) oppure un file stereo (canale sinistro contro destro)."""
    import librosa
    y, sr = sf.read(wav_a)
    if wav_b is None:
        a, b = y[:, 0], y[:, 1]
    else:
        a, b = y.mean(1), sf.read(wav_b)[0].mean(1)
    hop = 2048
    cL = librosa.feature.chroma_cqt(y=a.astype(np.float32), sr=sr, hop_length=hop)
    cR = librosa.feature.chroma_cqt(y=b.astype(np.float32), sr=sr, hop_length=hop)
    eL = librosa.feature.rms(y=a, hop_length=hop)[0][: cL.shape[1]]
    eR = librosa.feature.rms(y=b, hop_length=hop)[0][: cR.shape[1]]
    n = int(W * sr / hop)
    out = []
    soglia_e = 0.1 * max(np.percentile(eL, 95), np.percentile(eR, 95))
    for i in range(0, cL.shape[1] - n, n):
        if eL[i:i + n].mean() < soglia_e and eR[i:i + n].mean() < soglia_e:
            continue                                     # silenzio: non conta
        a, b = cL[:, i:i + n].mean(1), cR[:, i:i + n].mean(1)
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return np.array(out)


if __name__ == "__main__":
    for d in map(Path, sys.argv[1:]):
        s = similitudini(d / "stereo" / "chitarra_sinistra.wav", d / "stereo" / "chitarra_centro_destra.wav")
        print(f"{d.name:28s} finestre {len(s):3d} | similitudine mediana {np.median(s):.2f}, 10° perc. "
              f"{np.percentile(s, 10):.2f} | diverse (< {SOGLIA}) {100 * (s < SOGLIA).mean():4.1f}%")
