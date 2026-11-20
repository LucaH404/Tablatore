"""Test del Tool 6 (verifica_tablatura)."""
import json
import subprocess
import sys
from pathlib import Path

import guitarpro as gp
import pytest

from app.tools.tablatura import crea_tablatura
from app.tools.verifica import controlla, verifica_tablatura, verifica_tablatura_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"
REALE = AUDIO / "reale"
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "gt_note.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


def scrivi(tmp_path, note):
    f = tmp_path / "note.json"
    f.write_text(json.dumps({"sorgente": "prova.mp3", "note": [dict(n, intensita=0.8) for n in note]}), encoding="utf-8")
    return f


def ritmo(tmp_path, bpm=100.0, battute=8):
    beat = [round(i * 60 / bpm, 4) for i in range(battute * 4 + 1)]
    f = tmp_path / "ritmo.json"
    f.write_text(json.dumps({"bpm": bpm, "metro": "4/4", "beat": beat, "downbeat": beat[::4]}), encoding="utf-8")
    return f


def tablatura(tmp_path, note, **kw):
    return crea_tablatura(scrivi(tmp_path, note), ritmo(tmp_path, **kw), "standard", tmp_path)


def eventi(file_json):
    tab = json.loads(Path(file_json).read_text(encoding="utf-8"))
    return tab, [ev for b in tab["battute"] for ev in b["eventi"] if ev["note"]]


@pytest.fixture
def note_gt():
    return json.loads((AUDIO / "gt_note.json").read_text())["note"]


def test_tablatura_corretta_non_si_tocca(tmp_path, note_gt):
    r = tablatura(tmp_path, note_gt)
    v = verifica_tablatura(r.file_tablatura)
    assert v.suonabile and v.problemi_iniziali == 0 and v.note_tolte == 0 and v.note_riditeggiate == 0
    assert v.cicli == 0 and v.numero_note == r.numero_note
    assert eventi(v.file_tablatura)[1] == eventi(r.file_tablatura)[1]
    assert Path(r.file_gp5).is_file()                              # l'originale resta


def test_errori_fisici_trovati_e_corretti(tmp_path, note_gt):
    r = tablatura(tmp_path, note_gt)
    tab, ev = eventi(r.file_tablatura)
    a, b, c = ev[0]["note"][0], ev[1]["note"][0], ev[2]["note"][0]
    ev[0]["note"].append(dict(a, midi=a["midi"] + 5, tasto=a["tasto"] + 5))          # stessa corda
    b["tasto"] = 30                                                                  # oltre il manico
    ev[2]["note"].append(dict(c, corda=c["corda"] % 6 + 1, tasto=c["tasto"] + 9))    # mano troppo aperta
    (tmp_path / "rotta.json").write_text(json.dumps(tab), encoding="utf-8")
    tipi = {p.tipo for p in controlla(tab)}
    assert {"stessa_corda", "tasto_fuori_manico", "mano_troppo_aperta"} <= tipi
    v = verifica_tablatura(tmp_path / "rotta.json")
    assert v.suonabile and v.cicli >= 1 and v.problemi_finali == 0
    assert v.note_tolte == 0                       # le altezze erano suonabili: solo spostate
    # le corde e i tasti tornano coerenti con l'altezza
    corde = [64, 59, 55, 50, 45, 40]
    for e in eventi(v.file_tablatura)[1]:
        for n in e["note"]:
            assert corde[n["corda"] - 1] + n["tasto"] == n["midi"]


def test_nota_impossibile_tolta(tmp_path, note_gt):
    r = tablatura(tmp_path, note_gt)
    tab, ev = eventi(r.file_tablatura)
    ev[3]["note"].append({"corda": 6, "tasto": -10, "midi": 30, "nome": "F#1"})      # sotto il Mi grave
    (tmp_path / "rotta.json").write_text(json.dumps(tab), encoding="utf-8")
    v = verifica_tablatura(tmp_path / "rotta.json")
    assert v.suonabile and v.note_tolte == 1


def test_nota_isolata_acuta_tolta_e_buco_riempito(tmp_path):
    # arpeggio grave in crome con un'armonica spuria tre ottave sopra in mezzo
    note = [{"inizio": k * 0.3, "fine": k * 0.3 + 0.28, "midi": m}
            for k, m in enumerate([40, 47, 52, 47] * 4)]
    note.append({"inizio": 2.25, "fine": 2.4, "midi": 84})
    r = tablatura(tmp_path, note, battute=2)
    v = verifica_tablatura(r.file_tablatura)
    assert v.note_tolte == 1 and any("C6" in c for c in v.correzioni)
    tab, ev = eventi(v.file_tablatura)
    assert max(n["midi"] for e in ev for n in e["note"]) == 52
    assert all(sum(e["durata_slot"] for e in b["eventi"]) == b["slot_battuta"] for b in tab["battute"])
    pause = [e for b in tab["battute"] for e in b["eventi"] if not e["note"]]
    assert all(p["durata_slot"] >= 4 for p in pause)               # niente pause da una semicroma
    assert len(gp.parse(v.file_gp5).tracks[0].measures) == 2


def test_assolo_acuto_non_e_sospetto(tmp_path):
    # power chord gravi seguiti da una frase acuta di più note: nessuna nota è isolata
    note = [{"inizio": k * 0.3, "fine": k * 0.3 + 0.28, "midi": m} for k, m in enumerate([40, 40, 40, 40])]
    note += [{"inizio": 1.2 + k * 0.15, "fine": 1.33 + k * 0.15, "midi": m}
             for k, m in enumerate([76, 79, 81, 79, 76, 74, 76, 79])]
    r = tablatura(tmp_path, note, battute=1)
    v = verifica_tablatura(r.file_tablatura)
    assert v.note_tolte == 0


def test_salto_rapido_segnalato_non_corretto(tmp_path):
    # tablatura scritta a mano: dal 1° al 15° tasto e ritorno in una semicroma (150 ms)
    tab = {"accordatura": "standard", "corde": ["E2", "A2", "D3", "G3", "B3", "E4"], "bpm": 100, "metro": "4/4",
           "battute": [{"numero": 1, "slot_battuta": 16, "eventi": [
               {"slot": 0, "durata_slot": 1, "tempo_s": 0.0, "note": [{"corda": 6, "tasto": 1, "midi": 41, "nome": "F2"}]},
               {"slot": 1, "durata_slot": 1, "tempo_s": 0.15, "note": [{"corda": 1, "tasto": 15, "midi": 79, "nome": "G5"}]},
               {"slot": 2, "durata_slot": 14, "tempo_s": 0.3, "note": [{"corda": 6, "tasto": 1, "midi": 41, "nome": "F2"}]},
           ]}]}
    f = tmp_path / "salto.json"
    f.write_text(json.dumps(tab), encoding="utf-8")
    v = verifica_tablatura(f, correggi=False)
    assert any(p["tipo"] == "salto_rapido" for p in v.problemi)
    assert v.suonabile and any("salti" in a for a in v.avvisi)


def test_errori_di_ingresso(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"accordatura": "inventata", "battute": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        verifica_tablatura(f)


def test_tool_restituisce_riassunto_compatto(tmp_path, note_gt):
    r = tablatura(tmp_path, note_gt)
    testo = verifica_tablatura_audio(r.file_tablatura)
    dati = json.loads(testo)
    assert dati["suonabile"] and Path(dati["file_gp5"]).is_file()
    assert len(testo) < 1500


@pytest.mark.skipif(not (REALE / "gt_nine_thou_b7_12.json").exists(), reason="audio reale non presente")
def test_oracolo_nine_thou_nessuna_nota_vera_tolta(tmp_path):
    """Note esatte di Songsterr (con il riff che salta tra power chord gravi e note un'ottava
    sopra): il Verificatore non deve togliere niente."""
    from valuta_tablatura import riferimenti_nine_thou
    rif = riferimenti_nine_thou("Nine-thou-reale.mp3")
    beat = [3.965 + i * 60 / 108 for i in range(60)]
    f = tmp_path / "ritmo.json"
    f.write_text(json.dumps({"bpm": 108, "metro": "4/4", "beat": beat, "downbeat": beat[::4]}), encoding="utf-8")
    r = crea_tablatura(scrivi(tmp_path, rif), f, cartella_output=tmp_path)
    v = verifica_tablatura(r.file_tablatura)
    assert v.suonabile and v.note_tolte == 0 and v.problemi_iniziali == 0
