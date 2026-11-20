"""Esperimento Step 20: l'Ispettore (PANNs Cnn14_DecisionLevelMax, AudioSet, probabilità per frame).

Per ogni file stampa una linea del tempo a finestre di 0,5 s con le classi che interessano
(chitarra, batteria, synth, scratch...) e il riepilogo sul brano intero.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_ispettore.py <file audio> [altri file] [--da 0 --a 20]
"""
import sys
import time
from pathlib import Path

import numpy as np

CLASSI = {
    "chitarra el.": "Electric guitar", "chitarra": "Guitar", "distorsione": "Distortion",
    "batteria": "Drum kit", "drum machine": "Drum machine", "synth": "Synthesizer", "sampler": "Sampler",
    "scratch": "Scratching (performance technique)", "hip hop": "Hip hop music", "elettronica": "Electronic music",
    "basso": "Bass guitar",
}
FINESTRA_S = 0.5


def analizza(file_audio, sed):
    import librosa
    from panns_inference import labels
    y, _ = librosa.load(str(file_audio), sr=32000, mono=True)
    t0 = time.perf_counter()
    fw = sed.inference(y[None, :])[0]                # (frame, 527), 100 frame al secondo
    dt = time.perf_counter() - t0
    idx = {k: labels.index(v) for k, v in CLASSI.items()}
    fps = fw.shape[0] / (len(y) / 32000)
    n = int(len(y) / 32000 / FINESTRA_S)
    finestre = np.array([[fw[int(i * FINESTRA_S * fps):int((i + 1) * FINESTRA_S * fps), j].mean() for j in idx.values()]
                         for i in range(n)])
    return list(idx), finestre, dt, len(y) / 32000


def main(files, da, a):
    from panns_inference import SoundEventDetection
    sed = SoundEventDetection(checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"),
                              device="cuda")
    for f in files:
        nomi, w, dt, durata = analizza(f, sed)
        print(f"\n=== {Path(f).name}  ({durata:.0f} s, analisi {dt:.1f} s)")
        print("media sul brano: " + ", ".join(f"{k} {v:.2f}" for k, v in zip(nomi, w.mean(axis=0))))
        print("   t(s)  " + " ".join(f"{k[:9]:>9s}" for k in nomi))
        for i in range(int(da / FINESTRA_S), min(int(a / FINESTRA_S), len(w))):
            print(f"{i * FINESTRA_S:7.1f}  " + " ".join(f"{v:9.2f}" for v in w[i]))


if __name__ == "__main__":
    args = sys.argv[1:]
    da = float(args[args.index("--da") + 1]) if "--da" in args else 0.0
    a = float(args[args.index("--a") + 1]) if "--a" in args else 14.0
    files = [x for i, x in enumerate(args) if not x.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    main(files, da, a)
