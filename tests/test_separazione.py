"""Test del Tool 2 (separa_chitarra) sul brano sintetico con tracce note.

Richiede i file generati da scripts/genera_mix_test.py (vengono creati se mancano).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.tools.audio_io import carica_audio, salva_wav
from app.tools.separazione import separa_chitarra, separa_chitarra_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "mix.mp3").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12))


def test_mix_isola_la_chitarra(tmp_path):
    r = separa_chitarra(AUDIO / "mix.mp3", tmp_path)
    assert r.chitarra_presente and r.era_un_mix
    assert r.audio_consigliato == r.audio_chitarra
    assert Path(r.audio_chitarra).is_file()
    # La traccia isolata deve somigliare molto di più alla chitarra originale del mix intero
    gt = sf.read(AUDIO / "gt_chitarra.wav", dtype="float32")[0]
    stimata = carica_audio(r.audio_chitarra, 44100, canali=1)[0]
    mix = carica_audio(AUDIO / "mix.mp3", 44100, canali=1)[0]
    assert sdr(gt, stimata) > 5.0            # misurato: 7.18 dB (shifts=0)
    assert sdr(gt, stimata) - sdr(gt, mix) > 10.0   # misurato: +14.3 dB rispetto al mix


def test_solo_chitarra_usa_l_originale(tmp_path):
    r = separa_chitarra(AUDIO / "solo_chitarra.mp3", tmp_path)
    assert r.chitarra_presente and not r.era_un_mix
    accomp = sum(r.quote_energia[t] for t in ("drums", "bass", "vocals", "piano"))
    assert accomp < 0.15                     # misurato: 6%
    assert Path(r.audio_consigliato).name == "originale.wav"


def test_risultato_ripetibile(tmp_path):
    a = separa_chitarra(AUDIO / "mix.mp3", tmp_path)
    b = separa_chitarra(AUDIO / "mix.mp3", tmp_path)
    assert a.quote_energia == b.quote_energia


def test_brano_senza_chitarra(tmp_path):
    x = sf.read(AUDIO / "gt_basso.wav", dtype="float32")[0] + sf.read(AUDIO / "gt_batteria.wav", dtype="float32")[0]
    f = salva_wav(tmp_path / "senza_chitarra.wav", np.stack([x, x]), 44100)
    r = separa_chitarra(f, tmp_path)
    assert not r.chitarra_presente
    assert any("assente" in a for a in r.avvisi)


def test_durata_conservata(tmp_path):
    r = separa_chitarra(AUDIO / "mix.mp3", tmp_path)
    info = sf.info(r.audio_chitarra)
    assert abs(info.duration - r.durata_s) < 0.05
    assert info.samplerate == 44100


def test_tool_datapizza_restituisce_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # l'output di default va in ./output
    assert "percorso" in separa_chitarra_audio.schema["parameters"]["properties"]
    dati = json.loads(separa_chitarra_audio(str(AUDIO / "mix.mp3")))
    assert dati["chitarra_presente"] is True
    assert Path(dati["audio_consigliato"]).is_file()
