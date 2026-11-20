"""Esperimento: le due chitarre stanno in punti diversi dello stereo? (Dream On: base + melodia)

Sulla chitarra isolata (stereo) calcola per ogni punto tempo-frequenza la posizione nel panorama
(0 = tutto a sinistra, 0,5 = centro, 1 = tutto a destra), ne fa l'istogramma pesato per energia,
nel brano intero e per tratti, e divide la traccia in sinistra / centro / destra con maschere
(tipo ADRess, senza modelli). Scrive gli MP3 in <cartella>/stereo/.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_stereo.py <cartella_lavoro> [inizio fine ...]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIMITI = (0.4, 0.6)        # sotto: sinistra; sopra: destra; in mezzo: centro
TAGLIO_HZ = 80             # sotto il Mi grave della chitarra (82 Hz): solo basso rimasto nella traccia


def main(d: Path, tratti):
    import librosa
    y, sr = sf.read(d / "chitarra.wav")
    L, R = y[:, 0], y[:, 1]
    corr = np.corrcoef(L, R)[0, 1]
    eL, eR = (L ** 2).sum(), (R ** 2).sum()
    print(f"energia sinistra {10 * np.log10(eL / (eL + eR)):.1f} dB del totale, destra "
          f"{10 * np.log10(eR / (eL + eR)):.1f} dB | correlazione L/R {corr:.2f}")
    SL, SR = librosa.stft(L, n_fft=4096, hop_length=1024), librosa.stft(R, n_fft=4096, hop_length=1024)
    aL, aR = np.abs(SL), np.abs(SR)
    pan = aR / (aL + aR + 1e-9)
    peso = aL ** 2 + aR ** 2
    t = librosa.frames_to_time(np.arange(pan.shape[1]), sr=sr, hop_length=1024)
    bins = np.linspace(0, 1, 11)

    def istogramma(mask_t):
        h, _ = np.histogram(pan[:, mask_t].ravel(), bins=bins, weights=peso[:, mask_t].ravel())
        return h / (h.sum() + 1e-12)

    print("posizione nello stereo (0 = sinistra, 0,5 = centro, 1 = destra), quota di energia per decimo:")
    print("               " + " ".join(f"{b:.1f}" .rjust(5) for b in bins[:-1]))
    tutti = np.ones_like(t, dtype=bool)
    for nome, m in [("brano intero", tutti)] + [(f"{a:.0f}-{b:.0f} s", (t >= a) & (t < b)) for a, b in tratti]:
        h = istogramma(m)
        print(f"{nome:>14s} " + " ".join(f"{100 * x:4.0f}%" for x in h))
    out = d / "stereo"
    out.mkdir(exist_ok=True)
    freq = librosa.fft_frequencies(sr=sr, n_fft=4096)
    sopra = (freq >= TAGLIO_HZ)[:, None]
    for nome, m in (("sinistra", pan < LIMITI[0]), ("centro", (pan >= LIMITI[0]) & (pan <= LIMITI[1])),
                    ("destra", pan > LIMITI[1]), ("centro_destra", pan >= LIMITI[0])):
        m = m & sopra
        parte = np.stack([librosa.istft(SL * m, hop_length=1024, length=len(L)),
                          librosa.istft(SR * m, hop_length=1024, length=len(R))], axis=1)
        wav = out / f"chitarra_{nome}.wav"
        sf.write(wav, parte, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-b:a", "192k", str(wav.with_suffix(".mp3"))],
                       check=True)
        print(f"{nome}: {10 * np.log10((parte ** 2).sum() / (eL + eR) + 1e-12):.1f} dB rispetto alla chitarra intera")


if __name__ == "__main__":
    args = sys.argv[2:]
    main(Path(sys.argv[1]), [(float(args[i]), float(args[i + 1])) for i in range(0, len(args), 2)])
