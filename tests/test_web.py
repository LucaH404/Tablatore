"""Test della pagina web (FastAPI), senza LLM: costo 0."""
import json
import subprocess
import sys
import time
from pathlib import Path

import guitarpro as gp
import pytest
from fastapi.testclient import TestClient

import app.web.server as server
from app import config

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests" / "audio"


@pytest.fixture(scope="module", autouse=True)
def audio_di_prova():
    if not (AUDIO / "mix.mp3").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "genera_mix_test.py")], check=True)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    server.LAVORI = tmp_path_factory.mktemp("lavori")
    return TestClient(server.app)


def carica(client, nome="mix.mp3", contenuto=None, **campi):
    contenuto = contenuto if contenuto is not None else (AUDIO / "mix.mp3").read_bytes()
    dati = {"accordatura": "auto", "sensibilita": "auto", "resoconto_llm": "false", **campi}
    return client.post("/api/lavori", files={"file": (nome, contenuto, "audio/mpeg")}, data=dati)


def aspetta(client, id_lavoro, timeout=120):
    passi_visti = set()
    for _ in range(int(timeout / 0.2)):
        s = client.get(f"/api/lavori/{id_lavoro}").json()
        if s.get("passo"):
            passi_visti.add(s["passo"])
        if s["stato"] in ("finito", "errore"):
            return s, passi_visti
        time.sleep(0.2)
    raise TimeoutError


@pytest.fixture(scope="module")
def lavoro(client):
    r = carica(client, nome="../../Il mio brano!.mp3")
    assert r.status_code == 200
    return r.json()["id"], *aspetta(client, r.json()["id"])


def test_pagina_e_opzioni(client):
    risp = client.get("/")
    pagina = risp.text
    assert risp.headers["cache-control"] == "no-store"
    assert "Tablatore AI" in pagina and f"alphatab@{config.ALPHATAB_VERSIONE}" in pagina and "{{" not in pagina
    op = client.get("/api/opzioni").json()
    assert op["accordature"][0] == "auto" and "drop C#" in op["accordature"]
    assert op["corde"]["drop C#"] == [37, 44, 49, 54, 58, 63]
    for f in ("app.js", "esporta.js", "stile.css"):
        assert client.get(f"/static/{f}").status_code == 200
    assert "disegnaSuCanvas" in client.get("/stampa").text


def test_lavoro_completo(lavoro):
    id_lavoro, stato, passi = lavoro
    assert stato["stato"] == "finito"
    r = stato["risultato"]
    assert r["completato"] and r["accordatura"] == "standard" and r["bpm"] == 100.0
    assert stato["titolo"] == "Il mio brano"                 # nome ripulito: niente "../" né "!"
    assert r["avvisi"][-1].startswith("Le tablature di brani protetti")
    assert not any(k in r for k in ("file_gp5", "cartella", "file_traccia"))   # niente percorsi del PC
    assert set(r["file"]) >= {"tablatura_verificata.gp5", "note.mid", "traccia.json", "chitarra.mp3"}
    assert len(passi) >= 2                                   # l'avanzamento si è visto


def test_file_scaricabili(client, lavoro):
    id_lavoro, stato, _ = lavoro
    risp = client.get(stato["risultato"]["file"]["tablatura_verificata.gp5"])
    assert risp.status_code == 200 and "Il%20mio%20brano%20-%20tablatura.gp5" in risp.headers["content-disposition"]
    tmp = server.LAVORI / "scaricato.gp5"
    tmp.write_bytes(risp.content)
    assert len(gp.parse(str(tmp)).tracks[0].measures) == stato["risultato"]["numero_battute"]
    mp3 = client.get(stato["risultato"]["file"]["chitarra.mp3"])
    assert mp3.status_code == 200 and mp3.headers["content-type"] == "audio/mpeg" and mp3.content[:3] in (b"ID3", b"\xff\xfb")
    assert "Il%20mio%20brano%20-%20chitarra.mp3" in mp3.headers["content-disposition"]


def test_nessun_accesso_ad_altri_file(client, lavoro):
    id_lavoro, _, _ = lavoro
    for nome in ("risultato.json", "Il mio brano.mp3", "..%2F..%2Fconfig.py", "chitarra.wav"):
        assert client.get(f"/api/lavori/{id_lavoro}/file/{nome}").status_code == 404
    assert client.get("/api/lavori/..%2F..%2Fapp/file/note.mid").status_code == 404
    assert client.get("/api/lavori/123/file/note.mid").status_code == 404
    assert client.get(f"/api/lavori/{'0' * 32}").status_code == 404


def test_cambio_accordatura(client, lavoro):
    id_lavoro, _, _ = lavoro
    s = client.post(f"/api/lavori/{id_lavoro}/accordatura", data={"accordatura": "drop D"}).json()
    r = s["risultato"]
    assert r["accordatura"] == "drop D" and any("drop D" in d for d in r["decisioni"])
    tab = json.loads(client.get(r["file"]["tablatura_verificata.json"]).content)
    assert tab["accordatura"] == "drop D"
    assert client.post(f"/api/lavori/{id_lavoro}/accordatura", data={"accordatura": "inventata"}).status_code == 400
    r = client.post(f"/api/lavori/{id_lavoro}/accordatura",
                    data={"accordatura": "standard", "ripetizioni": "true"}).json()["risultato"]
    assert r["ripetizioni"]["attiva"] and r["separatore"] == "demucs"
    r = client.post(f"/api/lavori/{id_lavoro}/accordatura",
                    data={"accordatura": "standard", "ripetizioni": "false", "tempo": "0.5",
                          "semplifica": "true"}).json()["risultato"]
    assert r["bpm"] == 50.0 and r["fattore_tempo"] == 0.5 and r["semplifica"]["attiva"]
    assert client.post(f"/api/lavori/{id_lavoro}/accordatura",
                       data={"accordatura": "standard", "tempo": "3"}).status_code == 400
    client.post(f"/api/lavori/{id_lavoro}/accordatura",
                data={"accordatura": "standard", "tempo": "1", "semplifica": "false"})


def test_lavoro_ritrovato_dopo_riavvio(client, lavoro):
    id_lavoro, _, _ = lavoro
    server._lavori.clear()                                    # come se il server fosse ripartito
    s = client.get(f"/api/lavori/{id_lavoro}").json()
    assert s["stato"] == "finito" and s["risultato"]["completato"]


def test_estensione_e_parametri_rifiutati(client):
    assert carica(client, nome="brano.wav").status_code == 400
    assert carica(client, accordatura="inventata").status_code == 400
    assert carica(client, sensibilita="massima").status_code == 400
    assert carica(client, stile="metal").status_code == 400
    assert carica(client, separatore="spleeter").status_code == 400
    op = client.get("/api/opzioni").json()
    assert op["stili"] == ["auto", "power_chord", "libero"] and op["separatori"] == ["demucs", "roformer"]
    assert op["fattori_tempo"] == [0.5, 1.0, 2.0] and "4d semplifica" in op["passi"]


def test_banco_suoni_locale_o_ripiego(client, monkeypatch, tmp_path):
    """Step 23: la pagina usa il banco locale se c'è, altrimenti (None) quello piccolo del CDN."""
    monkeypatch.setattr(server, "STATICI", tmp_path)
    assert client.get("/api/opzioni").json()["banco_suoni"] is None
    (tmp_path / "suoni").mkdir()
    (tmp_path / "suoni" / config.BANCO_SUONI).write_bytes(b"RIFF")
    assert client.get("/api/opzioni").json()["banco_suoni"] == f"/static/suoni/{config.BANCO_SUONI}"


def test_file_troppo_grande(client, monkeypatch):
    monkeypatch.setattr(config, "DIMENSIONE_MAX_MB", 0.1)
    prima = set(server.LAVORI.iterdir())
    assert carica(client).status_code == 413
    assert set(server.LAVORI.iterdir()) == prima              # niente resti su disco


def test_file_finto_si_ferma_con_motivo(client):
    r = carica(client, nome="finto.mp3", contenuto=b"non sono un mp3")
    stato, _ = aspetta(client, r.json()["id"])
    assert stato["stato"] == "finito" and not stato["risultato"]["completato"]
    assert "non valido" in stato["risultato"]["motivo_stop"]


def test_tutor_endpoint(client, monkeypatch):
    """Step 25: /api/tutor passa domanda, storia e dati del brano al tutor; senza chiave risponde 503."""
    from app import tutor
    from app.tutor import RispostaTutor
    visto = {}

    def finto_chiedi(domanda, storia, brano):
        visto.update(domanda=domanda, storia=storia, brano=brano)
        return RispostaTutor(risposta="Risposta. [teoria/scale.md]", fonti=["teoria/scale.md"],
                             citate=["teoria/scale.md"], da_llm=True, costo_usd=0.0005)

    monkeypatch.setenv("OPENAI_API_KEY", "finta")
    monkeypatch.setattr(tutor, "chiedi", finto_chiedi)
    monkeypatch.setattr(tutor, "stato_indice", lambda: {"pronto": True})
    r = client.post("/api/tutor", json={"domanda": "Che scala uso?",
                                        "storia": [{"ruolo": "utente", "testo": "ciao"}]})
    assert r.status_code == 200 and r.json()["citate"] == ["teoria/scale.md"] and r.json()["da_llm"]
    assert visto["storia"] == [{"ruolo": "utente", "testo": "ciao"}] and visto["brano"] is None
    assert client.get("/api/opzioni").json()["tutor"] is True
    assert client.post("/api/tutor", json={"domanda": "x", "storia": [{"ruolo": "hacker", "testo": "x"}]}).status_code == 422
    assert client.post("/api/tutor", json={"domanda": "x", "id_lavoro": "../../etc"}).status_code == 404
    monkeypatch.delenv("OPENAI_API_KEY")
    r = client.post("/api/tutor", json={"domanda": "Che scala uso?"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
