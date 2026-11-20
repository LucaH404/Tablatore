"""Test del Tool 4 (analizza_ritmo): tempo, metro, battute e tonalità.

Tre casi con risposta nota: il mix sintetico (4/4, 100 BPM, Am-F-C-G), la sola chitarra
(senza batteria: il caso difficile) e un valzer sintetico (3/4, 140 BPM, Sol maggiore).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pytest

from app.tools.audio_io import salva_wav
from app.tools.ritmo import analizza_ritmo, analizza_ritmo_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"
BEAT_GT = np.arange(32) * 0.6          # 8 battute da 4 beat a 100 BPM
DOWNBEAT_GT = np.arange(8) * 2.4


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "gt_note.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


@pytest.fixture(scope="module")
def valzer(tmp_path_factory):
    """Valzer 3/4 a 140 BPM in Sol maggiore: cassa e basso sul 1, accordo sul 2 e sul 3."""
    sr, beat = 44100, 60 / 140
    y = np.zeros(int(sr * 22))
    rng = np.random.default_rng(1)

    def suono(freqs, dur, amp):
        t = np.arange(int(sr * dur)) / sr
        return amp * sum(np.sin(2 * np.pi * f * t) for f in freqs) * np.exp(-4 * t)

    def piazza(s, t):
        i = int(t * sr)
        y[i:i + len(s)] += s[: len(y) - i]

    accordi = [[196.0, 246.9, 293.7], [261.6, 329.6, 392.0], [293.7, 370.0, 440.0], [196.0, 246.9, 293.7]]
    bassi = [98.0, 130.8, 146.8, 98.0]          # G - C - D - G
    tk = np.arange(int(sr * 0.15)) / sr
    cassa = 0.8 * np.sin(2 * np.pi * (50 + 100 * np.exp(-30 * tk)) * tk) * np.exp(-12 * tk)
    th = np.arange(int(sr * 0.05)) / sr
    for b in range(int(20 / (3 * beat))):
        t0 = b * 3 * beat
        piazza(suono([bassi[b % 4]], 0.4, 0.5), t0)
        piazza(cassa, t0)
        for k in (1, 2):
            piazza(suono(accordi[b % 4], 0.3, 0.15), t0 + k * beat)
            piazza(0.1 * rng.uniform(-1, 1, len(th)) * np.exp(-60 * th), t0 + k * beat)
    y = np.stack([y, y]).astype(np.float32) / np.abs(y).max() * 0.8
    return salva_wav(tmp_path_factory.mktemp("valzer") / "valzer.wav", y, sr)


def f_beat(stimati, veri):
    return mir_eval.beat.f_measure(mir_eval.beat.trim_beats(veri), mir_eval.beat.trim_beats(np.array(stimati)))


def leggi(r):
    return json.loads(Path(r.file_ritmo).read_text(encoding="utf-8"))


def test_mix_4_4(tmp_path):
    r = analizza_ritmo(AUDIO / "mix.mp3", tmp_path)
    d = leggi(r)
    assert abs(r.bpm - 100) < 1, r.bpm                  # misurato: 100,0
    assert r.metro == "4/4"
    assert f_beat(d["beat"], BEAT_GT) > 0.9             # misurato: 0,96
    assert f_beat(d["downbeat"], DOWNBEAT_GT) > 0.85    # misurato: 0,91
    # Am-F-C-G: Do maggiore e La minore sono relative (stesse note, stessa armatura)
    assert {r.tonalita, r.tonalita_relativa} == {"C major", "A minor"}
    assert not r.avvisi


def test_sola_chitarra_senza_batteria(tmp_path):
    r = analizza_ritmo(AUDIO / "solo_chitarra.mp3", tmp_path)
    d = leggi(r)
    assert abs(r.bpm - 100) < 2, r.bpm                  # misurato: 99,8 (beat_this da solo: 130)
    assert r.metro == "4/4"
    assert f_beat(d["beat"], BEAT_GT) > 0.9             # misurato: 0,96 (beat_this da solo: 0,75)
    assert any("irregolari" in a for a in r.avvisi)     # l'utente viene avvisato
    assert {r.tonalita, r.tonalita_relativa} == {"C major", "A minor"}


def test_valzer_3_4(valzer, tmp_path):
    r = analizza_ritmo(valzer, tmp_path)
    d = leggi(r)
    assert abs(r.bpm - 140) < 1.4, r.bpm                # misurato: 140,0
    assert r.metro == "3/4"
    assert f_beat(d["downbeat"], np.arange(0, 20, 3 * 60 / 140)) > 0.85
    assert {r.tonalita, r.tonalita_relativa} == {"G major", "E minor"}


def test_silenzio(tmp_path):
    f = salva_wav(tmp_path / "silenzio.wav", np.zeros((2, 44100 * 5), dtype=np.float32), 44100)
    r = analizza_ritmo(f, tmp_path)
    assert r.bpm is None and r.metro is None and r.tonalita is None
    assert any("Ritmo non rilevato" in a for a in r.avvisi)


def test_beat_ordinati_dentro_il_brano(tmp_path):
    r = analizza_ritmo(AUDIO / "mix.mp3", tmp_path)
    d = leggi(r)
    assert np.all(np.diff(d["beat"]) > 0)
    assert 0 <= d["beat"][0] and d["beat"][-1] <= r.durata_s
    assert set(d["downbeat"]) <= set(d["beat"])         # ogni battuta inizia su un beat


def test_tool_restituisce_riassunto_compatto(tmp_path):
    copia = shutil.copy(AUDIO / "mix.mp3", tmp_path / "brano.mp3")
    testo = analizza_ritmo_audio(str(copia))
    dati = json.loads(testo)
    assert dati["bpm"] and Path(dati["file_ritmo"]).is_file()
    assert "beat" not in dati                           # l'elenco completo resta su file
    assert len(testo) < 1000
