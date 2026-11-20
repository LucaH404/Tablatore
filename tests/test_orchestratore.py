"""Test dell'Orchestratore (modalità codice, senza chiamate LLM: costo 0)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import app.orchestratore as orch
from app import config
from app.orchestratore import AVVISI_FISSI, _unisci_avvisi, numeri_inventati, trascrivi_brano

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "mix.mp3").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


@pytest.fixture(scope="module")
def completo(tmp_path_factory):
    out = tmp_path_factory.mktemp("brano")
    return trascrivi_brano(AUDIO / "mix.mp3", cartella_output=out, resoconto_llm=False), out


def test_pipeline_completa(completo):
    r, out = completo
    assert r.completato and r.motivo_stop is None
    assert r.accordatura == "standard" and r.bpm == 100.0 and r.metro == "4/4" and r.suonabile
    assert r.numero_battute >= 8 and r.numero_note > 50 and r.sensibilita == "media"
    assert Path(r.file_gp5).name == "tablatura_verificata.gp5" and Path(r.file_gp5).is_file()
    assert r.resoconto and "standard" in r.resoconto                  # testo di riserva, senza LLM
    assert r.token_in == 0 and r.costo_usd == 0
    # arpeggi puliti: la ricostruzione dei power chord NON deve attivarsi da sola (la regola su
    # questo brano porterebbe F1 da 0,88 a 0,17)
    assert r.power_chord["attiva"] is False and "non distorta" in r.power_chord["motivo"]


def test_tutti_i_file_in_una_cartella(completo):
    """O32: un brano = una cartella, per tutti i tool."""
    r, out = completo
    nomi = {f.name for f in out.iterdir()}
    assert {"chitarra.wav", "ritmo.json", "note.json", "note.mid", "tablatura.gp5", "tablatura_verificata.gp5",
            "traccia.json"} <= nomi
    assert all(Path(f).parent == out for f in (r.file_gp5, r.file_testo, r.file_midi, r.file_traccia))


def test_traccia_con_tutti_i_passi(completo):
    r, out = completo
    traccia = json.loads((out / "traccia.json").read_text(encoding="utf-8"))
    nomi = [s["nome"] for s in traccia["spans"] if s["nome"] != "passo 4b ottave"]   # solo con power chord spenti
    assert nomi[:7] == ["passo 1 validazione", "passo 2 separazione", "passo 3 ritmo", "passo 4 trascrizione",
                        "passo 4b power chord", "passo 5 tablatura", "passo 6 verifica"]
    assert all(s["esito"] == "ok" and s["durata_s"] >= 0 for s in traccia["spans"][:7])
    assert traccia["token_in"] == 0


def test_avvisi_raccolti_dal_codice(completo):
    """O27: gli avvisi dei tool arrivano all'utente senza passare dall'LLM; i fissi sempre in fondo."""
    r, _ = completo
    assert r.avvisi[-2:] == AVVISI_FISSI
    assert any("poco presente" in a for a in r.avvisi)            # avviso della separazione sul mix
    assert len(r.avvisi) == len(set(r.avvisi))


def test_unisci_avvisi_senza_doppioni():
    assert _unisci_avvisi(["a", AVVISI_FISSI[0]], ["a", "b"], None) == ["a", "b"] + AVVISI_FISSI


def test_sensibilita_auto_ripete_con_alta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NOTE_PER_BEAT_MIN", 50.0)       # forza "poche note per beat"
    r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=tmp_path, resoconto_llm=False)
    assert r.completato and r.sensibilita == "alta"
    assert r.decisioni and "sensibilità alta" in r.decisioni[0]
    assert any("decisione" in p for p in r.passi if p["nome"] == "passo 4 trascrizione")


def test_accordatura_indicata(tmp_path):
    r = trascrivi_brano(AUDIO / "mix.mp3", "drop D", cartella_output=tmp_path, resoconto_llm=False)
    assert r.completato and r.accordatura == "drop D"


def test_file_non_valido_si_ferma(tmp_path):
    f = tmp_path / "finto.mp3"
    f.write_text("non sono un mp3")
    r = trascrivi_brano(f, cartella_output=tmp_path / "out", resoconto_llm=False)
    assert not r.completato and r.motivo_stop.startswith("File non valido")
    assert [p["nome"] for p in r.passi] == ["passo 1 validazione"]
    assert r.avvisi[-2:] == AVVISI_FISSI


def test_brano_senza_chitarra_si_ferma(tmp_path):
    f = tmp_path / "batteria.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(AUDIO / "gt_batteria.wav"), "-b:a", "192k", str(f)],
                   check=True)
    r = trascrivi_brano(f, cartella_output=tmp_path / "out", resoconto_llm=False)
    assert not r.completato and "chitarra" in r.motivo_stop
    assert len(r.passi) == 2


def test_guasto_in_un_passo(tmp_path, monkeypatch):
    def rotto(*a, **k):
        raise RuntimeError("guasto simulato")
    monkeypatch.setattr(orch, "crea_tablatura", rotto)
    r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=tmp_path, resoconto_llm=False)
    assert not r.completato and "guasto simulato" in r.motivo_stop
    errati = [p for p in r.passi if p.get("esito") == "errore"]
    assert [p["nome"] for p in errati] == ["passo 5 tablatura"] and "guasto simulato" in errati[0]["errore"]
    assert Path(r.file_traccia).is_file()                          # la traccia c'è anche col guasto


def test_senza_chiave_resoconto_di_riserva(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(orch, "_client_llm", lambda: None)
    r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=tmp_path, resoconto_llm=True)
    assert r.completato and r.resoconto.startswith("Accordatura standard")
    assert "Resoconto automatico non disponibile" in r.avvisi[0]


def test_ripetizioni_passo_4c_e_riaccorda(tmp_path):
    r = trascrivi_brano(AUDIO / "mix.mp3", cartella_output=tmp_path, resoconto_llm=False, ripetizioni=True)
    assert r.completato and r.ripetizioni["attiva"] and r.separatore == "demucs"
    assert Path(r.file_note).name == "note_ripetizioni.json"
    assert "passo 4c ripetizioni" in [p["nome"] for p in r.passi]
    assert any(d.startswith("Confronto tra ripetizioni") for d in r.decisioni)
    spento = orch.riaccorda(r, "standard", ripetizioni=False)
    # senza ripetizioni si torna alle note del passo 4b (note_ottave.json se i power chord sono spenti, Step 22)
    assert not spento.ripetizioni["attiva"] and Path(spento.file_note).name in ("note.json", "note_ottave.json")
    assert not any(d.startswith("Confronto tra ripetizioni") for d in spento.decisioni)
    acceso = orch.riaccorda(spento, "standard", ripetizioni=True)
    assert acceso.ripetizioni["attiva"] and acceso.ripetizioni["note_aggiunte"] == r.ripetizioni["note_aggiunte"]


def test_tempo_e_semplifica(completo):
    """Step 19: con la batteria il tempo non si dimezza da solo; dimezzato a mano le battute sono la
    metà; "semplifica" (passo 4d) unisce i riattacchi e si spegne di nuovo."""
    r, out = completo
    assert r.fattore_tempo == 1.0 and r.bpm_rilevato == 100.0 and "tempo" not in r.avvisi_per_passo
    meta = orch.riaccorda(r, "standard", tempo=0.5, semplifica=True)
    assert meta.bpm == 50.0 and meta.fattore_tempo == 0.5 and meta.bpm_rilevato == 100.0
    assert abs(meta.numero_battute - r.numero_battute / 2) <= 1
    assert any(d.startswith("Tempo dimezzato: 100.0 → 50.0") for d in meta.decisioni)
    assert meta.semplifica["attiva"] and Path(meta.file_note).name == "note_semplificate.json"
    torna = orch.riaccorda(meta, "standard", tempo=1.0, semplifica=False)
    assert torna.bpm == 100.0 and torna.numero_battute == r.numero_battute and not torna.semplifica["attiva"]
    assert not any(d.startswith(("Tempo", "Note ripetute")) for d in torna.decisioni)
    with pytest.raises(ValueError):
        orch.riaccorda(r, "standard", tempo=3.0)


def test_parametri_non_validi(tmp_path):
    with pytest.raises(ValueError):
        trascrivi_brano(AUDIO / "mix.mp3", tempo=0.3, cartella_output=tmp_path)
    with pytest.raises(ValueError):
        trascrivi_brano(AUDIO / "mix.mp3", accordatura="inventata", cartella_output=tmp_path)
    with pytest.raises(ValueError):
        trascrivi_brano(AUDIO / "mix.mp3", sensibilita="massima", cartella_output=tmp_path)
    with pytest.raises(ValueError):
        trascrivi_brano(AUDIO / "mix.mp3", separatore="spleeter", cartella_output=tmp_path)


def test_guardrail_numeri_inventati():
    dati = {"bpm": 107.9, "battute": 12, "note_in_tablatura": 136, "metro": "4/4", "avvisi": ["chitarra al 12.4%"]}
    assert not numeri_inventati("Circa 108 BPM (107,9), 12 battute in 4/4, 136 note, chitarra al 12.4%.", dati)
    assert not numeri_inventati("Ci sono due salti e 3 punti da controllare.", dati)
    assert numeri_inventati("Il brano ha 16 battute.", dati) == {16.0}
    assert numeri_inventati("Suona il Do a 120 BPM.", dati) == {120.0}
