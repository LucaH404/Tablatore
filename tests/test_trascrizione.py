"""Test del Tool 3 (trascrivi_note) contro la partitura di riferimento del brano sintetico."""
import json
import subprocess
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import pytest

from app.tools.audio_io import salva_wav
from app.tools.separazione import separa_chitarra
from app.tools.trascrizione import nome_nota, trascrivi_note, trascrivi_note_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "gt_note.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


def f1_attacchi(file_note):
    gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
    est = json.loads(Path(file_note).read_text(encoding="utf-8"))["note"]
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        np.array([[n["inizio"], n["fine"]] for n in gt]),
        mir_eval.util.midi_to_hz(np.array([n["midi"] for n in gt])),
        np.array([[n["inizio"], n["fine"]] for n in est]),
        mir_eval.util.midi_to_hz(np.array([n["midi"] for n in est])),
        offset_ratio=None)
    return p, r, f


def test_nome_nota():
    assert nome_nota(60) == "C4" and nome_nota(40) == "E2" and nome_nota(69) == "A4"


def test_solo_chitarra(tmp_path):
    r = trascrivi_note(AUDIO / "solo_chitarra.mp3", cartella_output=tmp_path)
    p, rec, f = f1_attacchi(r.file_note)
    assert f > 0.90, (p, rec, f)              # misurato: 0.96
    assert rec > 0.95                         # misurato: 1.00
    assert r.nota_piu_grave == "C3"           # Do3, la nota più grave (accordo C)
    # Note con altezze che nel brano non esistono (armoniche residue): al massimo il 5%
    altezze_gt = {n["midi"] for n in json.loads((AUDIO / "gt_note.json").read_text())["note"]}
    est = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    estranee = [n for n in est if n["midi"] not in altezze_gt]
    assert len(estranee) <= 0.05 * len(est)   # misurato: 2 su 69 (due Sol5, armonica del Do4)
    assert r.polifonia_max <= 6


def test_mix_separato(tmp_path):
    sep = separa_chitarra(AUDIO / "mix.mp3", tmp_path)
    r = trascrivi_note(sep.audio_consigliato, cartella_output=tmp_path)
    p, rec, f = f1_attacchi(r.file_note)
    assert f > 0.80, (p, rec, f)              # misurato: 0.88


def test_file_midi_valido(tmp_path):
    r = trascrivi_note(AUDIO / "solo_chitarra.mp3", cartella_output=tmp_path)
    midi = pretty_midi.PrettyMIDI(r.file_midi)
    assert sum(len(i.notes) for i in midi.instruments) == r.numero_note


def test_sensibilita_ordinata(tmp_path):
    n = {s: trascrivi_note(AUDIO / "solo_chitarra.mp3", s, tmp_path / s).numero_note
         for s in ("bassa", "media", "alta")}
    assert n["bassa"] <= n["media"] <= n["alta"]


def test_silenzio_nessuna_nota(tmp_path):
    f = salva_wav(tmp_path / "silenzio.wav", np.zeros((2, 44100 * 5), dtype=np.float32), 44100)
    r = trascrivi_note(f, cartella_output=tmp_path)
    assert r.numero_note == 0
    assert any("Nessuna nota" in a for a in r.avvisi)


def test_sensibilita_non_valida(tmp_path):
    with pytest.raises(ValueError):
        trascrivi_note(AUDIO / "solo_chitarra.mp3", "massima", tmp_path)


def test_tool_restituisce_riassunto_compatto(tmp_path):
    import shutil
    copia = shutil.copy(AUDIO / "solo_chitarra.mp3", tmp_path / "brano.mp3")
    testo = trascrivi_note_audio(str(copia))
    dati = json.loads(testo)
    assert dati["numero_note"] > 0 and Path(dati["file_note"]).is_file()
    assert "note" not in dati                 # l'elenco completo resta su file
    assert len(testo) < 1000                  # pochi token per l'LLM
