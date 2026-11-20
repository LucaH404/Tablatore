"""Test del Tool 5 (crea_tablatura)."""
import json
import subprocess
import sys
from pathlib import Path

import guitarpro as gp
import numpy as np
import pytest

from app.tools.tablatura import crea_tablatura, crea_tablatura_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"
REALE = AUDIO / "reale"
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "gt_note.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


def scrivi(tmp_path, note, nome="note.json"):
    f = tmp_path / nome
    f.write_text(json.dumps({"sorgente": "prova.mp3",
                             "note": [dict(n, intensita=n.get("intensita", 0.8)) for n in note]}), encoding="utf-8")
    return f


def ritmo(tmp_path, bpm=100.0, battute=8, metro=4, ritardo=0.0, nome="ritmo.json"):
    beat = [round(i * 60 / bpm + ritardo, 4) for i in range(battute * metro + 1)]
    f = tmp_path / nome
    f.write_text(json.dumps({"bpm": bpm, "metro": f"{metro}/4", "beat": beat, "downbeat": beat[::metro]}),
                 encoding="utf-8")
    return f


def note_tab(r):
    tab = json.loads(Path(r.file_tablatura).read_text(encoding="utf-8"))
    return tab, [dict(n, tempo_s=ev["tempo_s"]) for b in tab["battute"] for ev in b["eventi"] for n in ev["note"]]


@pytest.fixture
def note_gt():
    return json.loads((AUDIO / "gt_note.json").read_text())["note"]


def test_oracolo_brano_di_prova(tmp_path, note_gt):
    from valuta_tablatura import confronta
    r = crea_tablatura(scrivi(tmp_path, note_gt), ritmo(tmp_path), cartella_output=tmp_path)
    tab, tn = note_tab(r)
    assert r.accordatura == "standard" and r.numero_battute == 8 and r.metro == "4/4"
    posto, precisione, _ = confronta(note_gt, tn)
    assert posto == 1.0 and precisione == 1.0
    assert r.note_scartate == 0 and not r.avvisi


def test_durate_battute_e_gp5(tmp_path, note_gt):
    r = crea_tablatura(scrivi(tmp_path, note_gt), ritmo(tmp_path), cartella_output=tmp_path)
    tab, _ = note_tab(r)
    assert all(sum(ev["durata_slot"] for ev in b["eventi"]) == b["slot_battuta"] for b in tab["battute"])
    s = gp.parse(r.file_gp5)
    assert s.tempo == 100 and len(s.tracks[0].measures) == 8
    assert [x.value for x in s.tracks[0].strings] == [64, 59, 55, 50, 45, 40]
    for m in s.tracks[0].measures:
        assert abs(sum(1 / b.duration.value * (1.5 if b.duration.isDotted else 1) for b in m.voices[0].beats) - 1) < 1e-9
    assert "|" in Path(r.file_testo).read_text(encoding="utf-8")


def test_calibrazione_di_fase(tmp_path, note_gt):
    """Beat in anticipo di 60 ms (come beat_this su "Nine thou"): le note restano al loro posto."""
    from valuta_tablatura import confronta
    r = crea_tablatura(scrivi(tmp_path, note_gt), ritmo(tmp_path, ritardo=-0.06), cartella_output=tmp_path)
    assert r.errore_quantizzazione_ms < 10
    assert confronta(note_gt, note_tab(r)[1])[0] == 1.0


def test_power_chord_a_un_dito_in_drop_c_diesis(tmp_path):
    # riff 0-7-7-5-8-7 di power chord (fondamentale, quinta, ottava) in drop C#
    note = []
    for k, tasto in enumerate([0, 7, 7, 5, 8, 7]):
        for corda_vuota in (37, 44, 49):
            note.append({"inizio": k * 0.3, "fine": k * 0.3 + 0.25, "midi": corda_vuota + tasto})
    r = crea_tablatura(scrivi(tmp_path, note), ritmo(tmp_path, bpm=100, battute=2), "drop C#", tmp_path)
    tab, tn = note_tab(r)
    for ev in [ev for b in tab["battute"] for ev in b["eventi"] if ev["note"]]:
        assert sorted(n["corda"] for n in ev["note"]) == [4, 5, 6]      # tre corde gravi
        assert len({n["tasto"] for n in ev["note"]}) == 1                # stesso tasto: un dito


def test_ottave_su_quinta_e_terza_corda(tmp_path):
    """Step 20 (O70): dopo un power chord 2-2-2, le ottave F3+F4, C#3+C#4, D#3+D#4 del ritornello di
    Lying From You vanno su 5a e 3a corda (9-11, 5-7, 7-9) come in Songsterr, non vicino al capotasto."""
    note = [{"inizio": 0.0, "fine": 0.25, "midi": m} for m in (39, 46, 51)]
    for k, basso in enumerate([53, 53, 49, 49, 51, 51]):
        note += [{"inizio": 0.3 * (k + 1), "fine": 0.3 * (k + 1) + 0.25, "midi": m} for m in (basso, basso + 12)]
    r = crea_tablatura(scrivi(tmp_path, note), ritmo(tmp_path, bpm=100, battute=2), "drop C#", tmp_path)
    tab, _ = note_tab(r)
    ottave = [sorted((n["corda"], n["tasto"]) for n in ev["note"]) for b in tab["battute"] for ev in b["eventi"]
              if len(ev["note"]) == 2]
    assert ottave == [[(3, 11), (5, 9)]] * 2 + [[(3, 7), (5, 5)]] * 2 + [[(3, 9), (5, 7)]] * 2


def test_accordatura_automatica_drop_c_diesis(tmp_path):
    # Do#2 (37) non esiste in accordatura standard: con molte note così (8 su 24) serve
    # un'accordatura ribassata
    note = [{"inizio": k * 0.3, "fine": k * 0.3 + 0.25, "midi": m} for k, m in enumerate([37, 44, 49] * 8)]
    r = crea_tablatura(scrivi(tmp_path, note), ritmo(tmp_path, battute=2), cartella_output=tmp_path)
    assert r.accordatura == "drop C#" and r.note_scartate == 0


def test_poche_note_false_non_cambiano_accordatura(tmp_path, note_gt):
    # brano in standard + 2 note spurie sotto il Mi grave (es. residui di basso): resta standard
    spurie = [{"inizio": 1.0, "fine": 1.2, "midi": 38}, {"inizio": 7.0, "fine": 7.2, "midi": 37}]
    r = crea_tablatura(scrivi(tmp_path, note_gt + spurie), ritmo(tmp_path), cartella_output=tmp_path)
    assert r.accordatura == "standard" and r.note_scartate == 2
    assert any("accordatura" in a for a in r.avvisi)


def test_note_impossibili_scartate_con_avviso(tmp_path):
    note = [{"inizio": 0.0, "fine": 0.5, "midi": 30}, {"inizio": 0.6, "fine": 1.1, "midi": 52}]
    r = crea_tablatura(scrivi(tmp_path, note), ritmo(tmp_path, battute=1), "standard", tmp_path)
    assert r.note_scartate == 1 and r.numero_note == 1
    assert any("scartate" in a for a in r.avvisi)


def test_battuta_in_levare(tmp_path):
    # una nota mezza battuta prima del primo tempo
    note = [{"inizio": 0.0, "fine": 0.3, "midi": 52}, {"inizio": 1.2, "fine": 1.5, "midi": 55}]
    beat = [1.2 + i * 0.6 for i in range(9)]
    f = tmp_path / "ritmo.json"
    f.write_text(json.dumps({"bpm": 100, "metro": "4/4", "beat": beat, "downbeat": beat[::4]}), encoding="utf-8")
    r = crea_tablatura(scrivi(tmp_path, note), f, "standard", tmp_path)
    tab, tn = note_tab(r)
    assert r.numero_battute == 2
    assert tab["battute"][1]["eventi"][0]["note"][0]["midi"] == 55       # la 2a battuta inizia sul primo tempo


def test_errori_di_ingresso(tmp_path, note_gt):
    with pytest.raises(ValueError):
        crea_tablatura(scrivi(tmp_path, note_gt), ritmo(tmp_path), "accordatura inventata", tmp_path)
    with pytest.raises(ValueError):
        crea_tablatura(scrivi(tmp_path, []), ritmo(tmp_path), cartella_output=tmp_path)


def test_tool_restituisce_riassunto_compatto(tmp_path, note_gt):
    testo = crea_tablatura_audio(str(scrivi(tmp_path, note_gt)), str(ritmo(tmp_path)))
    dati = json.loads(testo)
    assert Path(dati["file_gp5"]).is_file() and dati["numero_battute"] == 8
    assert len(testo) < 1200


@pytest.mark.skipif(not (REALE / "gt_nine_thou_b7_12.json").exists(), reason="audio reale non presente")
@pytest.mark.parametrize("nome", ["Nine thou - sintetizzato.mp3", "Nine-thou-reale.mp3"])
def test_oracolo_nine_thou(tmp_path, nome):
    """Note esatte di Songsterr + ritmo VERO di beat_this: drop C# riconosciuto, note al posto
    giusto, diteggiatura uguale a Songsterr nella maggior parte dei casi."""
    from valuta_tablatura import confronta, riferimenti_nine_thou
    from app.tools.ritmo import analizza_ritmo
    if not (REALE / nome).exists():              # gli MP3 sono esclusi da git: su un altro PC il test si salta
        pytest.skip(f"{nome} non presente")
    rif = riferimenti_nine_thou(nome)
    rit = analizza_ritmo(REALE / nome, tmp_path)
    r = crea_tablatura(scrivi(tmp_path, rif), rit.file_ritmo, cartella_output=tmp_path)
    posto, precisione, stessa_corda = confronta(rif, note_tab(r)[1])
    assert r.accordatura == "drop C#"
    assert posto >= 0.99 and precisione == 1.0
    assert stessa_corda > 0.8                     # misurato 83-84% (il resto: G#3 al 2° tasto invece del 12°)
