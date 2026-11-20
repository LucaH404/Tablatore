"""Valuta la trascrizione sulle battute 7-12 di "Nine thou" (riferimento: gt_nine_thou_b7_12.json).

Due tolleranze sull'attacco: 50 ms (standard) e 100 ms, perché la griglia del riferimento ha
un'incertezza di 50-100 ms nelle ultime battute (scarto crescente misurato sui picchi del modello).
Metriche: F1 esatto, F1 a meno dell'ottava; richiamo sul riff a note singole (b.7-8) e sugli
accordi (b.8-12, per nota); "accordi riconosciuti" = accordi con almeno 2 note giuste su 3.

Uso:  .venv\\Scripts\\python.exe scripts\\valuta_nine_thou_strofa.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.tools.separazione import separa_chitarra  # noqa: E402
from valuta_nine_thou import modello, run_inference, trascrivi  # noqa: E402

CARTELLA = ROOT / "tests" / "audio" / "reale"
RIF = json.loads((CARTELLA / "gt_nine_thou_b7_12.json").read_text(encoding="utf-8"))
hz = mir_eval.util.midi_to_hz


def valuta(est, rif, tol):
    da, a = min(n["inizio"] for n in rif) - 0.1, max(n["inizio"] for n in rif) + 0.1
    est = [(s, e, m) for s, e, m in est if da <= s <= a]
    if not est:
        return {"note": 0}
    gi = np.array([[n["inizio"], n["fine"]] for n in rif]); gp = np.array([n["midi"] for n in rif])
    ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est]); ep = np.array([m for *_, m in est])

    def prf(idx=None, ottava=False):
        g, p = (gi, gp) if idx is None else (gi[idx], gp[idx])
        if ottava:
            p, q = p % 12 + 60, ep % 12 + 60
        else:
            q = ep
        return mir_eval.transcription.precision_recall_f1_overlap(g, hz(p), ei, hz(q), onset_tolerance=tol, offset_ratio=None)

    singole = np.array([i for i, n in enumerate(rif) if n["battuta"] in (7, 8) and n["corda"] == 5 and
                        not any(m["inizio"] == n["inizio"] and m["corda"] != 5 for m in rif)])
    accordi_idx = np.array([i for i in range(len(rif)) if i not in set(singole)])
    # accordi riconosciuti: per ogni attacco di accordo (3 note), quante note trovate (esatte)
    trovate = 0; totale = 0
    attacchi = sorted({rif[i]["inizio"] for i in accordi_idx if sum(m["inizio"] == rif[i]["inizio"] for m in rif) == 3})
    for t in attacchi:
        totale += 1
        attese = [n["midi"] for n in rif if n["inizio"] == t]
        vicine = {m for s, _, m in est if abs(s - t) <= tol}
        trovate += len(set(attese) & vicine) >= 2
    return {"note": len(est), "F1": prf()[2], "F1_ottava": prf(ottava=True)[2],
            "rich_singole": prf(singole)[1], "rich_accordi": prf(accordi_idx)[1],
            "accordi_ok": f"{trovate}/{totale}"}


if __name__ == "__main__":
    out = ROOT / "output" / "reale" / "_strofa"
    for nome, note_rif in RIF["note"].items():
        chitarra = separa_chitarra(CARTELLA / nome, out / Path(nome).stem).audio_consigliato
        uscite = {"chitarra separata": run_inference(chitarra, modello),
                  "originale": run_inference(str(CARTELLA / nome), modello)}
        print(f"\n=== {nome}: battute 7-12, {len(note_rif)} note di riferimento ===")
        print("ingresso           sens.  toll. | note | F1   | F1 ott. | rich. singole | rich. accordi | accordi ok")
        for ingresso, u in uscite.items():
            for sens in ("media", "alta"):
                s = config.SENSIBILITA[sens]
                est = trascrivi(u, s["onset"], s["frame"], s["durata_min_ms"], config.FREQ_MIN_HZ, True)
                for tol in (0.05, 0.1):
                    r = valuta(est, note_rif, tol)
                    if r["note"] == 0:
                        print(f"{ingresso:18s} {sens:6s} {tol:.2f}  | 0 note"); continue
                    print(f"{ingresso:18s} {sens:6s} {tol:.2f}  | {r['note']:4d} | {r['F1']:.2f} | {r['F1_ottava']:.2f}    | "
                          f"{r['rich_singole']:.2f}          | {r['rich_accordi']:.2f}          | {r['accordi_ok']}")
