"""Test dello Step 22: Tool 4b-bis (note lette un'ottava sopra su chitarra pulita). Costo 0."""
import json

from app.tools.ottave import abbassa_ottave_sopra, correggi_ottave_sopra


def nota(t, midi, intensita=0.7, durata=0.3):
    return {"inizio": t, "fine": t + durata, "midi": midi, "intensita": intensita}


def arpeggio_con_f5():
    """Arpeggio C4-F4-G#4 ripetuto; ogni tanto il F4 esce come F5 debole."""
    note = []
    for k in range(6):
        t = k * 1.2
        note += [nota(t, 60), nota(t + 0.4, 65), nota(t + 0.8, 68, 0.75)]
    note.append(nota(3.0, 77, 0.5))                        # F5 debole e acuto: è il F4 un'ottava sopra
    return note


def test_abbassa_la_nota_debole_un_ottava_sopra():
    nuove, corrette = abbassa_ottave_sopra(arpeggio_con_f5())
    assert corrette == 1
    assert [n["midi"] for n in nuove].count(77) == 0 and nuove[-1]["midi"] == 65 and nuove[-1]["nome"] == "F4"


def test_non_tocca_una_nota_acuta_forte():
    note = arpeggio_con_f5()
    note[-1]["intensita"] = 0.9                             # più forte delle vicine: nota vera
    assert abbassa_ottave_sopra(note)[1] == 0


def test_non_tocca_senza_ottava_sotto_vicina():
    note = [nota(k * 0.5, 60) for k in range(10)] + [nota(2.0, 77, 0.4)]
    assert abbassa_ottave_sopra(note)[1] == 0


def test_scrive_note_ottave(tmp_path):
    f = tmp_path / "note.json"
    f.write_text(json.dumps({"note": arpeggio_con_f5(), "bpm": 78}), encoding="utf-8")
    r = correggi_ottave_sopra(f, tmp_path)
    dati = json.loads((tmp_path / "note_ottave.json").read_text(encoding="utf-8"))
    assert r.note_corrette == 1 and r.note_totali == 19 and dati["bpm"] == 78 and dati["ottave_corrette"] == 1
