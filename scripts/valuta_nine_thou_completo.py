"""Valutazione della trascrizione su tutto "Nine Thou" (Superstars Remix), con il riferimento
completo ricavato dalla tablatura Songsterr (scripts/riferimento_nine_thou_completo.py).

Metriche (note ghost escluse: sono stoppate, quasi senza altezza, e in metà strofe la chitarra
nel remix non si sente nemmeno):
  A. "quali note in quale tempo": per ogni beat si confronta l'insieme delle altezze attaccate
     (robusto ai dubbi di ritmo della tablatura). Anche a meno dell'ottava.
  B. attacco esatto (entro 70 ms) e altezza giusta, solo sui disegni con ritmo verificato.
  C. tablatura: per le note giuste, stessa corda e tasto di Songsterr?
Diagnosi: per ogni nota persa, che cosa c'era al suo posto (ottava sbagliata, semitono vicino,
niente); per ogni nota in più, che cos'è (armonica di una nota vera, altro).

Uso: .venv\\Scripts\\python.exe scripts\\valuta_nine_thou_completo.py <cartella_lavoro> [file_note]
     (cartella con chitarra.wav, ritmo.json, note.json di una trascrizione del brano intero)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

from riferimento_nine_thou_completo import riferimento_temporale  # noqa: E402

SEZIONI = {"intro1": "intro", "intro2": "intro", "riff": "riff", "fine": "riff", "riff_pause": "riff",
           "verso_a": "riff strofa", "verso_b": "riff strofa", "b21": "accordi con pause",
           "b22": "accordi con pause", "b23": "accordi con pause", "b26": "accordi con pause",
           "b48": "accordi con pause", "b49": "accordi con pause", "b50": "accordi con pause", "b51": "riff strofa",
           "b55": "accordi con pause", "pre_a": "pre-ritornello", "pre_b": "pre-ritornello",
           "pre_c": "pre-ritornello", "acc_0": "accordi tenuti", "acc_3": "accordi tenuti",
           "acc_6": "accordi tenuti", "b34": "pre-ritornello", "ritornello": "ritornello", "b39": "accordi tenuti",
           "b43": "riff strofa", "b76": "riff", "b81": "riff", "break_a": "breakdown", "break_b": "breakdown",
           "terzine": "riff", "finale": "accordi tenuti"}


def per_beat(note, confini, chiave):
    out = defaultdict(set)
    for n in note:
        out[n["beat"]].add(chiave(n))
    return out


def assegna_beat(note, confini):
    c = np.array(confini)
    for n in note:
        # un attacco poco prima del beat (fino a 35 ms) appartiene a quel beat
        n["beat"] = int(np.searchsorted(c, n["inizio"] + 0.035) - 1)
    return note


def valuta(rif, est, confini, etichetta=""):
    rif = [dict(n, beat=n["beat_globale"]) for n in rif if not n["ghost"]]
    est = assegna_beat([dict(n) for n in est], confini)
    r_b, e_b = per_beat(rif, confini, lambda n: n["midi"]), per_beat(est, confini, lambda n: n["midi"])
    r_pc, e_pc = per_beat(rif, confini, lambda n: n["midi"] % 12), per_beat(est, confini, lambda n: n["midi"] % 12)
    beats = set(r_b) | set(e_b)
    tp = sum(len(r_b[b] & e_b[b]) for b in beats)
    n_r, n_e = sum(len(v) for v in r_b.values()), sum(len(v) for v in e_b.values())
    tp_pc = sum(len(r_pc[b] & e_pc[b]) for b in beats)
    n_rpc, n_epc = sum(len(v) for v in r_pc.values()), sum(len(v) for v in e_pc.values())
    ris = {"richiamo": tp / n_r, "precisione": tp / max(n_e, 1), "richiamo_pc": tp_pc / n_rpc,
           "precisione_pc": tp_pc / max(n_epc, 1), "altezze_rif": n_r, "altezze_est": n_e}
    # B. attacco esatto sui disegni verificati
    ver = [n for n in rif if n["ritmo_verificato"]]
    t0, t1 = min(n["inizio"] for n in ver) - 1, max(n["inizio"] for n in ver) + 1
    usate, ok = set(), 0
    for n in ver:
        cand = [i for i, m in enumerate(est) if i not in usate and m["midi"] == n["midi"]
                and abs(m["inizio"] - n["inizio"]) <= 0.07]
        if cand:
            usate.add(min(cand, key=lambda i: abs(est[i]["inizio"] - n["inizio"])))
            ok += 1
    ris["attacco_esatto_richiamo"] = ok / len(ver)
    # diagnosi delle note perse e in più (per beat)
    perse, in_piu = Counter(), Counter()
    for b in beats:
        for m in r_b[b] - e_b[b]:
            if any((x - m) % 12 == 0 for x in e_b[b]):
                perse["ottava sbagliata"] += 1
            elif any(abs(x - m) == 1 for x in e_b[b]):
                perse["semitono vicino"] += 1
            elif b in e_b and e_b[b]:
                perse["altre note in quel tempo"] += 1
            else:
                perse["nessuna nota in quel tempo"] += 1
        for x in e_b[b] - r_b[b]:
            if any(x - m in (12, 24) for m in r_b[b]):
                in_piu["ottava sopra una nota vera (armonica)"] += 1
            elif any(x - m in (7, 19, 31) for m in r_b[b]):
                in_piu["quinta sopra una nota vera (armonica)"] += 1
            elif any(m - x in (12, 24) for m in r_b[b]):
                in_piu["ottava sotto una nota vera"] += 1
            elif not r_b.get(b):
                in_piu["tempo senza note nel riferimento"] += 1
            else:
                in_piu["altra nota"] += 1
    ris["perse"], ris["in_piu"] = dict(perse), dict(in_piu)
    # richiamo per corda (registro) e per sezione
    per_corda, per_sez = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for n in rif:
        trovata = n["midi"] in e_b.get(n["beat"], set())
        per_corda[n["corda"]][0] += trovata
        per_corda[n["corda"]][1] += 1
        s = SEZIONI.get(n["disegno"], n["disegno"])
        per_sez[s][0] += trovata
        per_sez[s][1] += 1
    ris["per_corda"] = {c: round(v[0] / v[1], 2) for c, v in sorted(per_corda.items())}
    ris["per_sezione"] = {s: f"{v[0] / v[1]:.0%} di {v[1]}" for s, v in per_sez.items()}
    return ris


def valuta_tablatura(rif, file_tablatura, confini):
    """C. Stessa corda e stesso tasto di Songsterr nello stesso tempo (quello che l'utente vede)."""
    tab = json.loads(Path(file_tablatura).read_text(encoding="utf-8"))
    note = assegna_beat([dict(n, inizio=ev["tempo_s"]) for b in tab["battute"] for ev in b["eventi"]
                         for n in ev["note"]], confini)
    r_tab, e_tab = defaultdict(set), defaultdict(set)
    for n in rif:
        if not n["ghost"]:
            r_tab[n["beat_globale"]].add((n["corda"], n["tasto"]))
    for n in note:
        e_tab[n["beat"]].add((n["corda"], n["tasto"]))
    giuste = sum(len(r_tab[b] & e_tab.get(b, set())) for b in r_tab)
    n_r, n_e = sum(len(v) for v in r_tab.values()), sum(len(v) for v in e_tab.values())
    return {"accordatura": tab["accordatura"], "richiamo": giuste / n_r, "precisione": giuste / max(n_e, 1)}


def stampa(nome, r):
    print(f"\n### {nome}")
    print(f"A. note giuste nel tempo giusto: richiamo {r['richiamo']:.0%}, precisione {r['precisione']:.0%} "
          f"({r['altezze_est']} altezze trascritte contro {r['altezze_rif']} nel riferimento)")
    print(f"   a meno dell'ottava:          richiamo {r['richiamo_pc']:.0%}, precisione {r['precisione_pc']:.0%}")
    print(f"B. attacco esatto (disegni con ritmo verificato): richiamo {r['attacco_esatto_richiamo']:.0%}")
    print(f"   note perse: {r['perse']}")
    print(f"   note in più: {r['in_piu']}")
    print(f"   richiamo per corda (6 = C#2 grave, 5 = quinta, 4 = ottava; 3-2 rare): {r['per_corda']}")
    print(f"   per sezione: {r['per_sezione']}")


if __name__ == "__main__":
    d = Path(sys.argv[1])
    rif, confini = riferimento_temporale(d / "chitarra.wav", d / "ritmo.json")
    forti = [n for n in rif if not n["ghost"]]
    print(f"riferimento: {len(rif)} note, {len(forti)} non ghost, "
          f"{sum(n['ritmo_verificato'] for n in forti)} con ritmo verificato")
    file_note = [Path(f) for f in sys.argv[2:]] or [f for f in (d / "note.json", d / "note_power_chord.json")
                                                     if f.exists()]
    for f in file_note:
        est = json.loads(f.read_text(encoding="utf-8"))["note"]
        stampa(f.name, valuta(rif, est, confini))
    if (d / "tablatura_verificata.json").exists():
        t = valuta_tablatura(rif, d / "tablatura_verificata.json", confini)
        print(f"\nC. tablatura ({t['accordatura']}): stessa corda e tasto di Songsterr nel tempo giusto per "
              f"{t['richiamo']:.0%} delle note di Songsterr ({t['precisione']:.0%} delle note in tablatura)")
