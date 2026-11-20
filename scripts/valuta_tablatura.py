"""Valuta il Tablatore (Tool 5).

Prova "oracolo": al Tablatore si danno le note ESATTE delle partiture di riferimento (come se la
trascrizione fosse perfetta), così si misura solo il Tablatore:
  - posto giusto: la nota finisce nella semicroma giusta con l'altezza giusta
  - accordatura scelta in automatico
  - stessa corda/tasto di Songsterr (informativo: altre diteggiature possono essere equivalenti)
Prova "reale": la pipeline completa (trascrizione 'alta' -> tablatura) sugli stessi brani.

Uso:  .venv\\Scripts\\python.exe scripts\\valuta_tablatura.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from app.tools.ritmo import analizza_ritmo  # noqa: E402
from app.tools.separazione import separa_chitarra  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402

AUDIO = ROOT / "tests" / "audio"
REALE = AUDIO / "reale"
TMP = Path(tempfile.mkdtemp())


def note_tab(file_json):
    tab = json.loads(Path(file_json).read_text(encoding="utf-8"))
    return tab, [dict(n, tempo_s=ev["tempo_s"]) for b in tab["battute"] for ev in b["eventi"]
                 if ev["note"] for n in ev["note"]]


def confronta(rif, tab_note, toll=0.07):
    """Per ogni nota di riferimento: c'è nella tablatura una nota con stessa altezza entro
    mezza semicroma? Se sì, è sulla stessa corda e tasto?"""
    usate, posto, corda = set(), 0, 0
    for n in rif:
        cand = [i for i, m in enumerate(tab_note) if i not in usate and m["midi"] == n["midi"]
                and abs(m["tempo_s"] - n["inizio"]) <= toll]
        if cand:
            i = min(cand, key=lambda i: abs(tab_note[i]["tempo_s"] - n["inizio"]))
            usate.add(i)
            posto += 1
            corda += ("corda" in n and tab_note[i]["corda"] == n["corda"] and tab_note[i]["tasto"] == n["tasto"])
    precisione = posto / len(tab_note) if tab_note else 0
    return posto / len(rif), precisione, corda / max(1, posto)


def scrivi_note(note, nome):
    f = TMP / f"{nome}_note.json"
    f.write_text(json.dumps({"sorgente": nome, "note": [dict(n, intensita=n.get("intensita", 0.8)) for n in note]}),
                 encoding="utf-8")
    return f


def riferimenti_nine_thou(nome):
    intro = json.loads((REALE / "gt_nine_thou_intro.json").read_text(encoding="utf-8"))
    strofa = json.loads((REALE / "gt_nine_thou_b7_12.json").read_text(encoding="utf-8"))
    t0 = intro["primo_tempo_s"][nome]
    return [dict(n, inizio=n["inizio"] + t0, fine=n["fine"] + t0) for n in intro["note"]] + strofa["note"][nome]


if __name__ == "__main__":
    print("=== Prova ORACOLO (note esatte) ===")
    gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
    ritmo = analizza_ritmo(AUDIO / "mix.mp3", TMP / "mix")
    r = crea_tablatura(scrivi_note(gt, "sintetico"), ritmo.file_ritmo, cartella_output=TMP / "t_sint")
    _, tn = note_tab(r.file_tablatura)
    rec, prec, _ = confronta(gt, tn)
    print(f"brano di prova: accordatura {r.accordatura}, {r.numero_battute} battute, ♩={r.bpm}, "
          f"posto giusto {rec:.0%} (precisione {prec:.0%}), aggiunte {r.note_aggiunte_power_chord}, "
          f"scarto quantizzazione {r.errore_quantizzazione_ms} ms, {r.tempo_elaborazione_s}s")
    if REALE.exists():
        for f in sorted(REALE.glob("*.mp3")):
            rif = riferimenti_nine_thou(f.name)
            rit = analizza_ritmo(f, TMP / f.stem)
            for pc, fase in ((False, True), (False, False), (True, True)):
                r = crea_tablatura(scrivi_note(rif, f.stem), rit.file_ritmo, cartella_output=TMP / f"o_{f.stem}_{pc}_{fase}",
                                   completa_power_chord=pc, calibra_fase=fase)
                _, tn = note_tab(r.file_tablatura)
                rec, prec, cor = confronta(rif, tn)
                print(f"{f.name} (power chord {'sì' if pc else 'no'}, fase {'sì' if fase else 'no'}): "
                      f"accordatura {r.accordatura}, {r.numero_battute} battute, posto giusto {rec:.0%} "
                      f"(precisione {prec:.0%}), stessa corda/tasto di Songsterr {cor:.0%}, "
                      f"scarto {r.errore_quantizzazione_ms} ms")
            r = crea_tablatura(scrivi_note(rif, f.stem), rit.file_ritmo, cartella_output=TMP / f"o_{f.stem}")
            print(Path(r.file_testo).read_text(encoding="utf-8")[:1500])

    print("\n=== Prova REALE (trascrizione 'alta' -> tablatura) ===")
    for f in sorted(REALE.glob("*.mp3")) if REALE.exists() else []:
        rif = riferimenti_nine_thou(f.name)
        rit = analizza_ritmo(f, TMP / f.stem)
        ch = separa_chitarra(f, TMP / "sep" / f.stem).audio_consigliato
        tr = trascrivi_note(ch, "alta", TMP / f"tr_{f.stem}", bpm=rit.bpm)
        for acc, pc, fase in (("auto", False, True), ("drop C#", False, True), ("drop C#", False, False),
                              ("drop C#", True, True)):
            if True:
                r = crea_tablatura(tr.file_note, rit.file_ritmo, acc, TMP / f"r_{f.stem}_{acc}_{pc}_{fase}",
                                   completa_power_chord=pc, calibra_fase=fase)
                _, tn = note_tab(r.file_tablatura)
                # solo le battute coperte dal riferimento
                t_min, t_max = min(n["inizio"] for n in rif) - 0.1, max(n["inizio"] for n in rif) + 0.1
                tn_rif = [n for n in tn if t_min <= n["tempo_s"] <= t_max]
                rec, prec, cor = confronta(rif, tn_rif)
                rec_ott = confronta([dict(n, midi=n["midi"] % 12) for n in rif],
                                    [dict(n, midi=n["midi"] % 12) for n in tn_rif])[0]
                print(f"{f.name} acc={acc:7s} pc={'sì' if pc else 'no'} fase={'sì' if fase else 'no'}: scelta {r.accordatura}, "
                      f"richiamo {rec:.0%} (a meno dell'ottava {rec_ott:.0%}), precisione {prec:.0%}, "
                      f"aggiunte {r.note_aggiunte_power_chord}, scartate {r.note_scartate}")
