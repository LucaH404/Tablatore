"""Esperimento Step 20 - passo B: beat calcolati sulla traccia della sola batteria invece che sul mix.

Per ogni cartella di lavoro (con originale.wav e ritmo.json): estrae la batteria con Demucs, ne misura
la quota di energia, rifà analizza_ritmo sulla batteria (in <cartella>/_batteria) e confronta con il
ritmo attuale: BPM, "senza batteria" (beat incoerenti), accordo tra i beat (F entro 70 ms).

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_tempo_batteria.py <cartella> [altre...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.tools.audio_io import carica_audio, salva_wav  # noqa: E402
from app.tools.ritmo import analizza_ritmo  # noqa: E402
from app.tools.separazione import _quote, _tracce_demucs  # noqa: E402


def f_beat(a, b, tol=0.07):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return 0.0
    ok = sum(np.abs(b - x).min() <= tol for x in a)
    return 2 * ok / (len(a) + len(b))


def main(cartelle):
    for d in cartelle:
        out = d / "_batteria"
        out.mkdir(exist_ok=True)
        sr = config.SAMPLE_RATE_LAVORO
        audio = carica_audio(d / "originale.wav", sr, canali=2)
        tracce, _ = _tracce_demucs(audio, sr)
        q = _quote(tracce)
        salva_wav(out / "batteria.wav", tracce["drums"], sr)
        rb = analizza_ritmo(out / "batteria.wav", out)
        mix = json.loads((d / "ritmo.json").read_text(encoding="utf-8"))
        bat = json.loads((out / "ritmo.json").read_text(encoding="utf-8"))
        # i beat a tempo doppio/metà: confronto anche con un beat sì e uno no
        f = max(f_beat(mix["beat"], bat["beat"]), f_beat(mix["beat"][::2], bat["beat"]),
                f_beat(mix["beat"], bat["beat"][::2]), f_beat(mix["beat"][1::2], bat["beat"]))
        print(f"{d.name:22s} quota batteria {q['drums']:.1%} | mix: {mix['bpm']} BPM, senza batteria "
              f"{mix.get('senza_batteria')} | batteria: {bat['bpm']} BPM, senza batteria {bat['senza_batteria']}, "
              f"metro {bat['metro']} | accordo beat F {f:.2f}")
        for a in rb.avvisi:
            print("     !", a)


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
