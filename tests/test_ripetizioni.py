"""Test del Tool 4c (vota_ripetizioni) e del separatore sperimentale RoFormer."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import app.tools.ripetizioni as rip
import app.tools.separazione as sep
from app.tools.ripetizioni import griglie, vota_ripetizioni

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"
BEAT = 1.0                                     # battuta da 4 s: tutto il riff sta in una battuta
RIFF = [(0.0, 37), (0.5, 44), (1.0, 49), (1.5, 44), (2.0, 37), (2.75, 40)]   # (posizione in s, midi)


def scrivi_caso(cartella: Path, battute=8):
    beat = [i * BEAT for i in range(4 * battute)]
    ritmo = {"beat": beat, "downbeat": beat[::4], "bpm": 120.0}
    note = []
    for b in range(battute):
        for pos, m in RIFF:
            if b == 3 and m == 49:              # nota persa in una ripetizione
                continue
            t = b * 4 * BEAT + pos
            note.append({"inizio": t, "fine": t + 0.2, "midi": m, "intensita": 0.6})
    note.append({"inizio": 5 * 4 * BEAT + 1.25, "fine": 5 * 4 * BEAT + 1.4, "midi": 70, "intensita": 0.3})  # falsa
    (cartella / "ritmo.json").write_text(json.dumps(ritmo), encoding="utf-8")
    (cartella / "note.json").write_text(json.dumps({"sorgente": "prova", "note": note}), encoding="utf-8")
    return cartella


def test_griglie_battute():
    gr = griglie({"beat": [i * 0.5 for i in range(16)], "downbeat": [0, 2, 4, 6]})
    assert len(gr) == 4 and np.allclose(gr[1], [2, 2.5, 3, 3.5, 4])


def test_ripetizioni_aggiungono_e_tolgono(tmp_path, monkeypatch):
    scrivi_caso(tmp_path)
    monkeypatch.setattr(rip, "_impronte", lambda f, gr, fmin: np.ones((len(gr), 4)) / 2)   # tutte uguali
    r = vota_ripetizioni(tmp_path / "note.json", tmp_path / "ritmo.json", tmp_path / "c.wav", tmp_path / "b.wav")
    assert r.battute == 8 and r.battute_con_ripetizioni == 8
    assert r.note_aggiunte == 1 and r.note_tolte == 1
    note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    b3 = [n for n in note if 12.0 <= n["inizio"] < 16.0]
    assert sorted(n["midi"] for n in b3) == sorted(m for _, m in RIFF)
    aggiunta = next(n for n in b3 if n.get("ripetizioni"))
    assert aggiunta["midi"] == 49 and abs(aggiunta["inizio"] - 13.0) < 1e-6     # al posto giusto
    assert not any(n["midi"] == 70 for n in note)
    assert Path(r.file_note).name == "note_ripetizioni.json"


def test_battute_tutte_diverse_non_cambiano(tmp_path, monkeypatch):
    scrivi_caso(tmp_path)
    monkeypatch.setattr(rip, "_impronte", lambda f, gr, fmin: np.eye(len(gr)))       # nessuna ripetizione
    r = vota_ripetizioni(tmp_path / "note.json", tmp_path / "ritmo.json", tmp_path / "c.wav", tmp_path / "b.wav")
    assert r.battute_con_ripetizioni == 0 and r.note_aggiunte == r.note_tolte == 0 and r.avvisi
    prima = json.loads((tmp_path / "note.json").read_text(encoding="utf-8"))["note"]
    dopo = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    assert len(prima) == len(dopo)


def test_basso_muto_non_conta(tmp_path):
    """Registrazione di sola chitarra: la traccia del basso è silenzio e non deve entrare nel confronto."""
    import soundfile as sf
    t = np.arange(22050) / 22050
    sf.write(tmp_path / "chitarra.wav", 0.3 * np.sin(2 * np.pi * 110 * t), 22050)
    sf.write(tmp_path / "muto.wav", 1e-7 * np.sin(2 * np.pi * 55 * t), 22050)
    sf.write(tmp_path / "basso.wav", 0.3 * np.sin(2 * np.pi * 55 * t), 22050)
    assert not rip._basso_udibile(tmp_path / "chitarra.wav", tmp_path / "muto.wav")
    assert rip._basso_udibile(tmp_path / "chitarra.wav", tmp_path / "basso.wav")


def test_brano_troppo_corto(tmp_path):
    scrivi_caso(tmp_path, battute=2)
    r = vota_ripetizioni(tmp_path / "note.json", tmp_path / "ritmo.json", tmp_path / "c.wav", tmp_path / "b.wav")
    assert r.file_note == str(tmp_path / "note.json") and "corto" in r.avvisi[0]


# ---------------------------------------------------------------- separatore RoFormer
@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "mix.mp3").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


def test_separatore_non_valido(tmp_path):
    with pytest.raises(ValueError):
        sep.separa_chitarra(AUDIO / "mix.mp3", tmp_path, separatore="spleeter")


def test_roformer_non_disponibile_ripiega_su_demucs(tmp_path, monkeypatch):
    def rotto(audio):
        raise OSError("pesi non scaricabili")
    monkeypatch.setattr(sep, "_tracce_roformer", rotto)
    r = sep.separa_chitarra(AUDIO / "mix.mp3", tmp_path, separatore="roformer")
    assert r.separatore == "demucs" and r.chitarra_presente
    assert any("RoFormer non disponibile" in a for a in r.avvisi)


def _pesi_roformer_presenti():
    try:
        import bs_roformer  # noqa: F401
        from bs_roformer import default_models_dir
        return any(Path(default_models_dir()).rglob("*.ckpt"))
    except Exception:
        return False


@pytest.mark.skipif(not _pesi_roformer_presenti(), reason="bs-roformer-infer o pesi non installati")
def test_roformer_senza_chitarra_ripiega_su_demucs(tmp_path):
    """Sul brano sintetico RoFormer mette la chitarra nel basso: si deve tornare a Demucs (O60)."""
    r = sep.separa_chitarra(AUDIO / "mix.mp3", tmp_path, separatore="roformer")
    assert r.separatore == "demucs" and r.chitarra_presente
    assert any("non ha trovato la chitarra" in a for a in r.avvisi)
