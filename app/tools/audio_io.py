"""Lettura e scrittura audio condivise tra i tool.

La decodifica passa da ffmpeg (già verificato dal Guardiano), così ogni tool riceve
lo stesso formato: float32, stereo, alla frequenza richiesta.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def carica_audio(percorso: str | Path, sample_rate: int, canali: int = 2) -> np.ndarray:
    """Decodifica qualsiasi file audio con ffmpeg. Ritorna un array (canali, campioni) float32."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg non trovato: installare ffmpeg e aggiungerlo al PATH")
    cmd = [exe, "-v", "error", "-i", str(percorso), "-f", "f32le", "-ac", str(canali),
           "-ar", str(sample_rate), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Decodifica fallita: {proc.stderr.decode(errors='ignore')[:200]}")
    dati = np.frombuffer(proc.stdout, dtype=np.float32)
    return dati.reshape(-1, canali).T.copy()


def salva_mp3(wav: str | Path, mp3: str | Path, kbps: int = 192) -> Path:
    """Converte un WAV in MP3 con ffmpeg (per ascoltare la traccia isolata, Step 20)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg non trovato")
    proc = subprocess.run([exe, "-y", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k",
                           str(mp3)], capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Conversione in MP3 fallita: {proc.stderr.decode(errors='ignore')[:200]}")
    return Path(mp3)


def salva_wav(percorso: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Salva un array (canali, campioni) come WAV float32."""
    p = Path(percorso)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, audio.T, sample_rate, subtype="FLOAT")
    return p
