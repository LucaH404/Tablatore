"""Esperimento: attacchi doppi (nota tenuta spezzata in più attacchi) - Step 19.

Per ogni accordo che ripete l'accordo precedente ancora in suono ("continuazione") si calcolano
indizi dall'audio e dalla griglia; su Nine Thou il riferimento Songsterr dice se l'attacco è vero.
Per ogni indizio: AUC (0,5 = inutile, 1 = separa perfettamente vere e false).

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_riattacchi.py <cartella> [nine]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from app.tools.power_chord import _gruppi  # noqa: E402


def inviluppi(file_audio):
    import librosa
    import soundfile as sf
    y, sr = sf.read(file_audio)
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=22050, hop_length=hop)
    return t, env[:len(t)], rms


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def indizi(d: Path):
    note = json.loads((d / "note_power_chord.json").read_text(encoding="utf-8"))["note"]
    gruppi = _gruppi(note)
    beat = np.array(json.loads((d / "ritmo.json").read_text(encoding="utf-8"))["beat"])
    t, env, rms = inviluppi(d / "chitarra.wav")
    righe = []
    for i in range(1, len(gruppi)):
        g, p = gruppi[i], gruppi[i - 1]
        t0, t1 = p[0]["inizio"], g[0]["inizio"]
        if {n["midi"] for n in g} != {n["midi"] for n in p} or max(n["fine"] for n in p) < t1 - 0.03:
            continue
        k = int(np.clip(np.searchsorted(beat, t1) - 1, 0, len(beat) - 2))
        pos = (t1 - beat[k]) / (beat[k + 1] - beat[k]) * 4          # in semicrome
        dur_beat = beat[k + 1] - beat[k]
        tra = (t > t0 + 0.03) & (t < t1 - 0.005)
        dopo = (t >= t1) & (t < t1 + 0.06)
        prima = (t >= t0) & (t < t0 + 0.06)
        picco = env[(t >= t1 - 0.03) & (t <= t1 + 0.03)].max()
        livello = rms[dopo].max() if dopo.any() else 1e-9
        righe.append(dict(
            t=t1,
            calo=float(rms[tra].min() / livello) if tra.any() else 1.0,           # basso = suono smorzato prima
            salita=float(livello / (rms[tra][-3:].mean() + 1e-9)) if tra.any() else 1.0,
            picco=float(picco / (np.median(env) + 1e-9)),
            griglia=float(abs(pos - round(pos))),                                   # 0 = sulla semicroma
            distanza=float((t1 - t0) / dur_beat),
            rapporto_inizio=float(livello / (rms[prima].max() + 1e-9)),
        ))
    return righe, len(gruppi)


def main(d: Path, nine: bool):
    righe, n = indizi(d)
    print(f"accordi {n}, continuazioni {len(righe)}")
    if not nine:
        return righe
    from riferimento_nine_thou_completo import riferimento_temporale
    rif, _ = riferimento_temporale(ROOT / "output/_banco/nine_thou/chitarra.wav", d / "ritmo.json")
    ta = np.array(sorted({x["inizio"] for x in rif}))
    vero = np.array([np.abs(ta - r["t"]).min() < 0.06 for r in righe])
    print(f"vere {vero.sum()}, false {(~vero).sum()}")
    for k in ("calo", "salita", "picco", "griglia", "distanza", "rapporto_inizio"):
        v = np.array([r[k] for r in righe])
        a = auc(v[vero], v[~vero])
        print(f"  {k:16s} AUC {a:.2f}  mediana vere {np.median(v[vero]):.3f} false {np.median(v[~vero]):.3f}")
    return righe, vero


if __name__ == "__main__":
    main(Path(sys.argv[1]), len(sys.argv) > 2)
