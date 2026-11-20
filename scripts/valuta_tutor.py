"""Valutazione del tutor (Step 25) sulle domande di tests/tutor_gold.json.

1. Recupero (solo embedding, costo ~0): recall@1, recall@k, MRR della scheda attesa; similitudine massima
   delle domande in tema e fuori tema (per calibrare config.TUTOR_SIMILITUDINE_MIN).
2. Risposte complete (con --risposte): esito dei guardrail, citazioni valide, scheda attesa citata,
   token, costo e tempo per domanda. Scrive il dettaglio in output/valutazione_tutor.json.

Uso: .venv\\Scripts\\python.exe scripts\\valuta_tutor.py [--risposte]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from app import config, tutor  # noqa: E402


def main(risposte: bool):
    gold = json.loads((ROOT / "tests" / "tutor_gold.json").read_text(encoding="utf-8"))
    emb = tutor._embedder()
    k = config.TUTOR_K
    righe, sim_in, sim_out = [], [], []
    t0 = time.perf_counter()
    for g in gold["in_tema"]:
        passaggi = tutor.cerca(g["domanda"], k=k, embedder=emb)
        fonti = list(dict.fromkeys(f for _, f, _ in passaggi))
        rango = fonti.index(g["fonte"]) + 1 if g["fonte"] in fonti else None
        sim_in.append(passaggi[0][0])
        righe.append({**g, "fonti": fonti, "rango": rango, "sim": round(passaggi[0][0], 3)})
    for d in gold["fuori_tema"]:
        sim_out.append(tutor.cerca(d, k=1, embedder=emb)[0][0])
    n = len(righe)
    r1 = sum(1 for r in righe if r["rango"] == 1) / n
    rk = sum(1 for r in righe if r["rango"]) / n
    mrr = sum(1 / r["rango"] for r in righe if r["rango"]) / n
    print(f"RECUPERO ({n} domande in tema, k={k}, {time.perf_counter() - t0:.1f} s)")
    print(f"   recall@1 {100 * r1:.0f}%   recall@{k} {100 * rk:.0f}%   MRR {mrr:.2f}")
    for r in righe:
        if r["rango"] != 1:
            print(f"   rango {r['rango']}: «{r['domanda']}» atteso {r['fonte']}, trovati {r['fonti']}")
    print(f"   similitudine massima: in tema min {min(sim_in):.3f} / mediana {statistics.median(sim_in):.3f}; "
          f"fuori tema max {max(sim_out):.3f} / mediana {statistics.median(sim_out):.3f}")
    print(f"   soglia attuale TUTOR_SIMILITUDINE_MIN = {config.TUTOR_SIMILITUDINE_MIN}: in tema rifiutate "
          f"{sum(s < config.TUTOR_SIMILITUDINE_MIN for s in sim_in)}, fuori tema accettate "
          f"{sum(s >= config.TUTOR_SIMILITUDINE_MIN for s in sim_out)}")
    out = {"recupero": {"recall@1": r1, f"recall@{k}": rk, "mrr": mrr, "sim_in": sim_in, "sim_out": sim_out,
                        "righe": righe}}
    if risposte:
        dettagli = []
        for g in gold["in_tema"] + [{"domanda": d, "fonte": None} for d in gold["fuori_tema"]]:
            r = tutor.chiedi(g["domanda"], embedder=emb)
            dettagli.append({"domanda": g["domanda"], "atteso": g["fonte"], "risposta": r.risposta, "fonti": r.fonti,
                             "citate": r.citate, "da_llm": r.da_llm, "rifiutata": r.rifiutata, "scartata": r.scartata,
                             "token_in": r.token_in, "token_out": r.token_out, "costo_usd": r.costo_usd,
                             "tempo_s": r.tempo_s})
        it = [d for d in dettagli if d["atteso"]]
        ft = [d for d in dettagli if not d["atteso"]]
        gen = [d for d in dettagli if d["da_llm"]]
        print(f"\nRISPOSTE ({len(dettagli)} domande)")
        print(f"   in tema: risposte dall'LLM {sum(d['da_llm'] for d in it)}/{len(it)}, scartate dal guardrail "
              f"{sum(bool(d['scartata']) for d in it)}, rifiutate {sum(bool(d['rifiutata']) for d in it)}, "
              f"con citazioni {sum(bool(d['citate']) for d in it)}, scheda attesa citata "
              f"{sum(d['atteso'] in d['citate'] for d in it)}")
        print(f"   fuori tema: rifiutate in ingresso {sum(bool(d['rifiutata']) for d in ft)}/{len(ft)}, "
              f"risposte dall'LLM {sum(d['da_llm'] for d in ft)}")
        if gen:
            print(f"   per risposta generata: token in mediana {statistics.median(d['token_in'] for d in gen):.0f}, "
                  f"out {statistics.median(d['token_out'] for d in gen):.0f}, costo medio "
                  f"${statistics.mean(d['costo_usd'] for d in gen):.5f}, tempo mediano "
                  f"{statistics.median(d['tempo_s'] for d in gen):.1f} s (max {max(d['tempo_s'] for d in gen):.1f})")
        print(f"   costo totale della valutazione ${sum(d['costo_usd'] for d in dettagli):.4f}")
        for d in dettagli:
            if d["scartata"] or (d["atteso"] and d["rifiutata"]) or (not d["atteso"] and d["da_llm"]):
                print(f"   ⚠ «{d['domanda']}»: rifiutata={d['rifiutata']} scartata={d['scartata']}")
        out["risposte"] = dettagli
    f = ROOT / "output" / "valutazione_tutor.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ndettaglio in {f.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main("--risposte" in sys.argv)
    finally:
        tutor.chiudi_indice()
