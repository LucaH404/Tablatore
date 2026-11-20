"""Test del Tool 4b (completa_power_chord): audio sintetico costruito nel test."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.tools.power_chord import completa_power_chord, completa_power_chord_audio, linea_di_basso, piattezza_spettrale

ROOT = Path(__file__).resolve().parents[1]
REALE = ROOT / "tests" / "audio" / "reale"
COMPLETO = REALE / "Nine Thou (Grant Mohrman Superstars Remix) - Styles Of Beyond (youtube).mp3"
# riferimento trascritto da Songsterr: escluso da git come l'MP3 (solo uso locale)
RIFERIMENTO = ROOT / "scripts" / "riferimento_nine_thou_completo.py"
SR = 22050
BPM_PASSO = 0.5                       # un attacco ogni mezzo secondo
RADICI = [37, 44, 45, 42, 37, 44, 45, 42] * 4      # C#2 G#2 A2 F#2 (riff in drop C#)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def scrivi_basso(f, radici):
    t = np.arange(int(len(radici) * BPM_PASSO * SR)) / SR
    y = np.concatenate([0.5 * np.sin(2 * np.pi * hz(r - 12) * t[:int(BPM_PASSO * SR)]) for r in radici])
    sf.write(f, y, SR)


def scrivi_chitarra(f, radici, distorta):
    t = np.arange(int(BPM_PASSO * SR)) / SR
    pezzi = []
    for r in radici:
        y = sum(np.sin(2 * np.pi * hz(m) * t) for m in (r, r + 7, r + 12)) / 3
        if distorta:
            y = np.clip(8 * y, -1, 1) + 0.05 * np.random.default_rng(r).standard_normal(len(t))
        pezzi.append(0.5 * y * np.exp(-2 * t))
    sf.write(f, np.concatenate(pezzi), SR)


def scrivi_note(f, gruppi):
    note = [{"inizio": i * BPM_PASSO, "fine": i * BPM_PASSO + 0.4, "midi": m, "intensita": 0.6}
            for i, g in enumerate(gruppi) for m in g]
    f.write_text(json.dumps({"sorgente": "prova", "note": note}), encoding="utf-8")
    return f


@pytest.fixture
def cartella(tmp_path):
    scrivi_basso(tmp_path / "basso.wav", RADICI)
    scrivi_chitarra(tmp_path / "chitarra.wav", RADICI, distorta=True)
    scrivi_chitarra(tmp_path / "pulita.wav", RADICI, distorta=False)
    # come Basic Pitch sulla chitarra distorta: niente fondamentale, solo quinta e/o ottava
    gruppi = [[r + 7, r + 12] if i % 3 == 0 else [r + 12] if i % 3 == 1 else [r + 7] for i, r in enumerate(RADICI)]
    scrivi_note(tmp_path / "note.json", gruppi)
    return tmp_path


def test_linea_di_basso(cartella):
    t, midi, quota = linea_di_basso(cartella / "basso.wav")
    assert quota > 0.9
    for i, r in enumerate(RADICI[:8]):
        sel = (t > i * BPM_PASSO + 0.1) & (t < i * BPM_PASSO + 0.4)
        assert int(np.nanmedian(midi[sel])) == r - 12
    assert (cartella / "linea_basso.npz").exists()          # salvata per non ricalcolarla


def test_piattezza_distingue_pulita_e_distorta(cartella):
    assert piattezza_spettrale(cartella / "pulita.wav") < 0.0003 < piattezza_spettrale(cartella / "chitarra.wav")


def test_power_chord_ricostruiti(cartella):
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav")
    assert r.attiva and r.attacchi_convertiti == len(RADICI) and "distorta" in r.motivo
    note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    for i, radice in enumerate(RADICI):
        alt = sorted(n["midi"] for n in note if abs(n["inizio"] - i * BPM_PASSO) < 0.01)
        assert alt == [radice, radice + 7, radice + 12]       # fondamentale, quinta, ottava


def test_chitarra_pulita_resta_spenta(cartella):
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "pulita.wav")
    assert not r.attiva and "non distorta" in r.motivo
    assert Path(r.file_note) == cartella / "note.json"         # note originali, nessun file nuovo


def test_note_incoerenti_con_il_basso_non_si_toccano(cartella):
    # terza minore sopra il basso (accordo minore arpeggiato): non è un power chord
    scrivi_note(cartella / "note.json", [[r + 3] for r in RADICI])
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav")
    assert not r.attiva and "coerenti" in r.motivo
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav", "power_chord")
    assert r.attiva and r.attacchi_convertiti == 0              # forzata, ma nessun attacco coerente


def test_stile_forzato_e_spento(cartella):
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "pulita.wav", "power_chord")
    assert r.attiva and r.avvisi                                # avviso: forzata su chitarra pulita
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav", "libero")
    assert not r.attiva and r.attacchi_convertiti == 0
    with pytest.raises(ValueError):
        completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav", "metal")


def test_basso_assente(cartella):
    sf.write(cartella / "muto.wav", np.zeros(SR * 16), SR)
    r = completa_power_chord(cartella / "note.json", cartella / "muto.wav", cartella / "chitarra.wav")
    assert r.quota_basso == 0.0 and "basso quasi assente" in r.motivo
    assert not r.attiva                                         # solo 11 accordi: troppo pochi per il ripiego


def test_senza_basso_fondamentali_dalla_chitarra(cartella):
    """Step 18: registrazione di sola chitarra. Accordi senza fondamentale (quinta + ottava), basso muto
    o assente: le fondamentali si deducono dalla chitarra stessa."""
    scrivi_note(cartella / "note.json", [[r + 7, r + 12] for r in RADICI])
    sf.write(cartella / "muto.wav", np.zeros(SR * 16), SR)
    for basso in (cartella / "muto.wav", None):
        r = completa_power_chord(cartella / "note.json", basso, cartella / "chitarra.wav")
        assert r.attiva and r.fonte == "chitarra" and "basso quasi assente" in r.motivo
        assert r.attacchi_convertiti == len(RADICI) and r.coerenza_interna == 1.0
        note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
        for i, radice in enumerate(RADICI):
            alt = sorted(n["midi"] for n in note if abs(n["inizio"] - i * BPM_PASSO) < 0.01)
            assert alt == [radice, radice + 7, radice + 12]
    # chitarra pulita: niente ripiego
    r = completa_power_chord(cartella / "note.json", None, cartella / "pulita.wav")
    assert not r.attiva and "non distorta" in r.motivo


def test_basso_incoerente_e_accordi_pieni(cartella):
    # basso che suona altro (residuo della chitarra): ripiego sulla chitarra
    scrivi_basso(cartella / "altro.wav", [r + 3 for r in RADICI])
    scrivi_note(cartella / "note.json", [[r + 7, r + 12] for r in RADICI])
    r = completa_power_chord(cartella / "note.json", cartella / "altro.wav", cartella / "chitarra.wav")
    assert r.attiva and r.fonte == "chitarra" and "poco coerenti con il basso" in r.motivo
    # accordi minori pieni (fondamentale, terza minore, quinta): non sono power chord, restano spenti
    scrivi_note(cartella / "note.json", [[r, r + 3, r + 7] for r in RADICI])
    r = completa_power_chord(cartella / "note.json", None, cartella / "chitarra.wav")
    assert not r.attiva and r.coerenza_interna == 0.0
    # con il basso coerente vince il basso
    scrivi_note(cartella / "note.json", [[r + 7, r + 12] for r in RADICI])
    r = completa_power_chord(cartella / "note.json", cartella / "basso.wav", cartella / "chitarra.wav")
    assert r.attiva and r.fonte == "basso"


def test_ottave_restano_ottave(cartella, monkeypatch):
    """Step 20 (O70): raffiche di ottave (F3+F4) senza basso restano ottave, anche le note singole
    della raffica; con il basso che suona la stessa nota sotto diventano power chord."""
    from app import config
    monkeypatch.setattr(config, "FINESTRA_OTTAVA_S", 0.6)    # qui gli attacchi distano 0,5 s (nel brano 0,19)
    gruppi = []
    for i, r in enumerate(RADICI):
        gruppi.append([r + 7, r + 12] if i % 4 else [53, 65])          # un'ottava ogni 4 attacchi...
    gruppi[1] = [53]                                                   # ...e una nota sola della raffica
    scrivi_note(cartella / "note.json", gruppi)
    r = completa_power_chord(cartella / "note.json", None, cartella / "chitarra.wav")
    assert r.attiva and r.fonte == "chitarra"
    note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    alt = lambda i: sorted(n["midi"] for n in note if abs(n["inizio"] - i * BPM_PASSO) < 0.01)
    assert alt(0) == [53, 65] and alt(1) == [53, 65]                    # ottava e nota della raffica
    assert alt(2) == [RADICI[2], RADICI[2] + 7, RADICI[2] + 12]         # il resto: power chord
    # basso che suona F sotto l'ottava: è un power chord di F (si sentiva solo l'ottava)
    scrivi_basso(cartella / "basso_f.wav", [53 if i % 4 == 0 else r for i, r in enumerate(RADICI)])
    scrivi_note(cartella / "note.json", [[r + 7, r + 12] if i % 4 else [53, 65] for i, r in enumerate(RADICI)])
    r = completa_power_chord(cartella / "note.json", cartella / "basso_f.wav", cartella / "chitarra.wav")
    assert r.fonte == "basso"
    note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    assert sorted(n["midi"] for n in note if abs(n["inizio"]) < 0.01) == [41, 48, 53]


def test_tool_restituisce_json(cartella):
    d = json.loads(completa_power_chord_audio(str(cartella / "note.json"), str(cartella / "basso.wav"),
                                              str(cartella / "chitarra.wav")))
    assert d["attiva"] and Path(d["file_note"]).is_file()


@pytest.mark.skipif(not (COMPLETO.exists() and RIFERIMENTO.exists()), reason="brano completo o riferimento non presente")
def test_nine_thou_completo(tmp_path):
    """Brano intero con riferimento Songsterr: in automatico i power chord si attivano, l'accordatura
    drop C# viene riconosciuta e la tablatura coincide con Songsterr molto più di prima (3%)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from riferimento_nine_thou_completo import riferimento_temporale
    from valuta_nine_thou_completo import valuta, valuta_tablatura

    from app.orchestratore import trascrivi_brano
    r = trascrivi_brano(COMPLETO, cartella_output=tmp_path, resoconto_llm=False)
    assert r.completato and r.power_chord["attiva"] and r.accordatura == "drop C#"
    rif, confini = riferimento_temporale(tmp_path / "chitarra.wav", tmp_path / "ritmo.json")
    note = json.loads(Path(r.file_note).read_text(encoding="utf-8"))["note"]
    v = valuta(rif, note, confini)
    assert v["richiamo"] > 0.65 and v["precisione"] > 0.65            # misurato 71% / 70%
    assert valuta_tablatura(rif, r.file_tablatura, confini)["richiamo"] > 0.58   # misurato 64%


def _roformer_pronto():
    try:
        from bs_roformer import default_models_dir
        return any(Path(default_models_dir()).rglob("*.ckpt"))
    except Exception:
        return False


@pytest.mark.skipif(not (COMPLETO.exists() and RIFERIMENTO.exists() and _roformer_pronto()), reason="brano completo, riferimento o RoFormer assente")
def test_nine_thou_roformer_e_ripetizioni(tmp_path):
    """Opzioni sperimentali (Step 17): RoFormer + confronto tra ripetizioni. Riferimento allineato
    sulla chitarra di Demucs, come nel banco di prova, per non cambiare il metro di giudizio."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from riferimento_nine_thou_completo import riferimento_temporale
    from valuta_nine_thou_completo import valuta_tablatura

    from app.orchestratore import trascrivi_brano
    r = trascrivi_brano(COMPLETO, cartella_output=tmp_path, resoconto_llm=False, separatore="roformer",
                        ripetizioni=True)
    assert r.completato and r.separatore == "roformer" and r.accordatura == "drop C#"
    base = ROOT / "output" / "_banco" / "nine_thou"
    cartella_rif = base if (base / "chitarra.wav").exists() else tmp_path
    rif, confini = riferimento_temporale(cartella_rif / "chitarra.wav", tmp_path / "ritmo.json")
    assert valuta_tablatura(rif, r.file_tablatura, confini)["richiamo"] > 0.70   # misurato 75% nel banco
