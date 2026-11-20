"""Test su audio reale: "Nine thou" (registrazioni fornite dall'utente, solo per studio).

Riferimento: prime 3 battute dell'intro dalla tablatura Songsterr (scripts/riferimento_nine_thou.py).
I file non sono distribuiti col progetto: se mancano, i test vengono saltati.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARTELLA = ROOT / "tests" / "audio" / "reale"
SINT = CARTELLA / "Nine thou - sintetizzato.mp3"
REALE = CARTELLA / "Nine-thou-reale.mp3"
pytestmark = pytest.mark.skipif(not (SINT.exists() and REALE.exists()
                                     and (CARTELLA / "gt_nine_thou_intro.json").exists()),
                                reason="audio reale non presente")

from app.tools.ritmo import analizza_ritmo  # noqa: E402
from app.tools.separazione import separa_chitarra  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402
from app.tools.validazione import valida_file  # noqa: E402


@pytest.fixture(scope="module")
def chitarre(tmp_path_factory):
    out = tmp_path_factory.mktemp("reale")
    return {f.name: separa_chitarra(f, out / f.stem).audio_consigliato for f in (SINT, REALE)}


def punteggio_intro(file_note, nome):
    sys.path.insert(0, str(ROOT / "scripts"))
    from valuta_nine_thou import RIF, valuta
    note = json.loads(Path(file_note).read_text(encoding="utf-8"))["note"]
    return valuta([(n["inizio"], n["fine"], n["midi"]) for n in note], RIF["primo_tempo_s"][nome])


@pytest.mark.parametrize("f, silenzio", [(SINT, 3.75), (REALE, 3.93)])
def test_validazione_vede_il_silenzio_iniziale(f, silenzio):
    r = valida_file(f)
    assert r.valido
    assert abs(r.metadati["silenzio_iniziale_s"] - silenzio) < 0.2
    assert any("all'inizio" in a for a in r.avvisi)


@pytest.mark.parametrize("f", [SINT, REALE])
def test_ritmo_108_bpm_in_4_4(f, tmp_path):
    r = analizza_ritmo(f, tmp_path)
    assert abs(r.bpm - 108) < 1.5, r.bpm             # Songsterr: 108; misurato 108,1 e 107,9
    assert r.metro == "4/4"
    assert abs(r.inizio_prima_battuta_s - (3.76 if f == SINT else 3.92)) < 0.1
    assert not any("irregolari" in a for a in r.avvisi)


def test_media_suggerisce_alta(chitarre, tmp_path):
    r = trascrivi_note(chitarre[SINT.name], "media", tmp_path, bpm=108)
    assert r.note_per_beat < 1                       # misurato: ~0,8
    assert any("sensibilita='alta'" in a for a in r.avvisi)


def test_alta_migliora_il_riff_sintetizzato(chitarre, tmp_path):
    f_media = punteggio_intro(trascrivi_note(chitarre[SINT.name], "media", tmp_path / "m").file_note, SINT.name)
    f_alta = punteggio_intro(trascrivi_note(chitarre[SINT.name], "alta", tmp_path / "a").file_note, SINT.name)
    assert f_alta[0] > f_media[0] + 0.15, (f_media, f_alta)   # misurato: 0,29 -> 0,53
    assert f_alta[3] > 0.8                                    # note acute trovate: 89%


def punteggio_strofa(file_note, nome, tol=0.1):
    sys.path.insert(0, str(ROOT / "scripts"))
    from valuta_nine_thou_strofa import RIF, valuta
    note = json.loads(Path(file_note).read_text(encoding="utf-8"))["note"]
    return valuta([(n["inizio"], n["fine"], n["midi"]) for n in note], RIF["note"][nome], tol)


@pytest.mark.skipif(not (CARTELLA / "gt_nine_thou_b7_12.json").exists(), reason="riferimento b.7-12 assente")
def test_strofa_reale_con_band_alta(chitarre, tmp_path):
    """Battute 7-12 della registrazione reale (riff + power chord distorti, con batteria e basso)."""
    media = punteggio_strofa(trascrivi_note(chitarre[REALE.name], "media", tmp_path / "m").file_note, REALE.name)
    alta = punteggio_strofa(trascrivi_note(chitarre[REALE.name], "alta", tmp_path / "a").file_note, REALE.name)
    assert media["F1"] < 0.3                          # misurato 0,23: le soglie del sintetico non bastano
    assert alta["F1"] > 0.4, alta                     # misurato 0,47 (0,59 a meno dell'ottava)
    ok, tot = map(int, alta["accordi_ok"].split("/"))
    assert ok / tot > 0.5                             # misurato 16/26 power chord riconosciuti


@pytest.mark.xfail(reason="Limite noto (O29): intro con phaser e filtro, Basic Pitch non "
                          "riconosce le altezze (confidenza < 0,5)", strict=False)
def test_intro_reale_con_effetti(chitarre, tmp_path):
    r = trascrivi_note(chitarre[REALE.name], "alta", tmp_path)
    assert punteggio_intro(r.file_note, REALE.name)[1] > 0.5
