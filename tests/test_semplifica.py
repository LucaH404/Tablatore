"""Test dello Step 19: tempo con fattore (ritmo_con_fattore) e Tool 4d (semplifica_note). Costo 0."""
import json

import pytest

from app.tools.ritmo import ritmo_con_fattore
from app.tools.semplifica import semplifica_note, unisci_note_tenute

BEAT = [i * 0.375 for i in range(33)]              # 160 BPM, 8 battute di 4/4


def accordo(t, radice, durata=0.2):
    return [{"inizio": t, "fine": t + durata, "midi": m, "intensita": 0.5} for m in (radice, radice + 7, radice + 12)]


def test_ritmo_dimezzato_e_raddoppiato(tmp_path):
    f = tmp_path / "ritmo.json"
    f.write_text(json.dumps({"bpm": 160.0, "metro": "4/4", "beat": BEAT, "downbeat": BEAT[1::4]}), encoding="utf-8")
    m = ritmo_con_fattore(f, 0.5, tmp_path / "meta.json")
    assert m["bpm"] == 80.0 and m["beat"] == BEAT[1::2] and m["downbeat"] == BEAT[1::8]
    assert set(m["downbeat"]) <= set(m["beat"]) and m["fattore_tempo"] == 0.5
    d = ritmo_con_fattore(f, 2, tmp_path / "doppio.json")
    assert d["bpm"] == 320.0 and len(d["beat"]) == 2 * len(BEAT) - 1 and d["downbeat"][0] == BEAT[1]
    assert d["downbeat"][1] == pytest.approx(BEAT[3])  # 4 beat nuovi = 2 vecchi
    assert ritmo_con_fattore(f, 1, tmp_path / "uguale.json")["beat"] == BEAT
    with pytest.raises(ValueError):
        ritmo_con_fattore(f, 3, tmp_path / "x.json")


def test_unisce_solo_i_riattacchi_mentre_suona():
    note = (accordo(0.0, 37, 0.25) + accordo(0.12, 37)          # frammento: stesso accordo, ancora in suono
            + accordo(0.75, 37, 0.1) + accordo(0.95, 37)        # riattacco dopo che il primo è finito: resta
            + accordo(1.5, 39, 0.3) + accordo(1.6, 37)          # accordo diverso: resta
            + accordo(2.25, 42, 0.5) + accordo(2.55, 42))       # stesso accordo ma oltre mezzo beat: resta
    nuove, uniti, prima = unisci_note_tenute(note, BEAT)
    assert (uniti, prima) == (1, 8)
    inizi = sorted({n["inizio"] for n in nuove})
    assert inizi == [0.0, 0.75, 0.95, 1.5, 1.6, 2.25, 2.55]
    assert all(n["fine"] == pytest.approx(0.32) for n in nuove if n["inizio"] == 0.0)   # il primo si allunga


def test_attacchi_diversi_nella_stessa_casella():
    """Due accordi diversi nella stessa semicroma: resta il più vicino, l'altro va nella casella accanto
    (o si scarta se è occupata); un riattacco con le stesse note resta unito."""
    import numpy as np

    from app.tools.tablatura import _risolvi_collisioni
    slot_t = np.arange(8) * 0.2
    def q(t, radice, slot):
        return [dict(n, slot=slot, durata_slot=2) for n in accordo(t, radice)]
    note = q(0.21, 37, 1) + q(0.28, 39, 1)          # stessa casella 1: C# vicino, D# dopo -> casella 2
    note += q(0.79, 42, 4) + q(0.85, 44, 4) + q(1.0, 46, 5)   # casella 4 piena e 5 occupata -> G# scartato
    note += q(1.39, 37, 7) + q(1.45, 37, 7)         # riattacco uguale: unito
    out = _risolvi_collisioni(note, slot_t)
    per_slot = {}
    for n in out:
        per_slot.setdefault(n["slot"], set()).add(n["midi"])
    assert per_slot[1] == {37, 44, 49} and per_slot[2] == {39, 46, 51}
    assert per_slot[4] == {42, 49, 54} and per_slot[5] == {46, 53, 58} and 44 not in per_slot[4]
    assert per_slot[7] == {37, 44, 49} and sum(1 for n in out if n["slot"] == 7) == 6


def test_semplifica_note_scrive_il_file(tmp_path):
    (tmp_path / "ritmo.json").write_text(json.dumps({"beat": BEAT}), encoding="utf-8")
    note = [n for k in range(10) for n in accordo(k * 0.75, 37, 0.3) + accordo(k * 0.75 + 0.1, 37)]
    (tmp_path / "note.json").write_text(json.dumps({"note": note}), encoding="utf-8")
    r = semplifica_note(tmp_path / "note.json", tmp_path / "ritmo.json")
    assert r.accordi_prima == 20 and r.accordi_uniti == 10 and r.avvisi           # metà uniti: avviso
    dati = json.loads((tmp_path / "note_semplificate.json").read_text(encoding="utf-8"))
    assert dati["semplificate"] and len(dati["note"]) == 30
