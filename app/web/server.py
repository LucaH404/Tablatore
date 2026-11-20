"""Pagina web locale: carica un MP3, segue l'avanzamento, mostra la tablatura con alphaTab.

Avvio:  .venv\\Scripts\\python.exe -m app.web.server   ->  http://127.0.0.1:8000

API (JSON):
  GET  /api/opzioni                     accordature e sensibilità disponibili
  POST /api/lavori                      carica un MP3 (multipart) -> {"id": ...}
  GET  /api/lavori/{id}                 stato, passo in corso, risultato
  POST /api/lavori/{id}/accordatura     rifà tablatura + verifica con un'altra accordatura
  GET  /api/lavori/{id}/file/{nome}     scarica un file del risultato (solo un elenco fisso)
  POST /api/tutor                       domanda al tutor di teoria (RAG): {domanda, storia, id_lavoro}

Sicurezza: il server ascolta solo su 127.0.0.1; il limite di dimensione è controllato mentre
il file arriva; il nome del file viene ripulito; gli id dei lavori sono casuali e controllati
con un'espressione regolare; si scaricano solo i file dell'elenco FILE_SCARICABILI (niente
percorsi scelti dall'utente). Un solo lavoro alla volta: i modelli usano la stessa GPU.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, tutor
from app.orchestratore import RisultatoBrano, riaccorda, trascrivi_brano
from app.tools.power_chord import MODALITA as STILI

ROOT = Path(__file__).resolve().parents[2]
STATICI = Path(__file__).parent / "static"
LAVORI = ROOT / config.CARTELLA_LAVORI
ID_VALIDO = re.compile(r"^[0-9a-f]{32}$")
PASSI = ["1 validazione", "2 separazione", "3 ritmo", "4 trascrizione", "4b power chord", "4b ottave", "4c ripetizioni",
         "4d semplifica", "5 tablatura", "6 verifica", "7 resoconto LLM"]
FILE_SCARICABILI = {
    "tablatura_verificata.gp5": "application/octet-stream",
    "tablatura_verificata.txt": "text/plain; charset=utf-8",
    "tablatura_verificata.json": "application/json",
    "note.mid": "audio/midi",
    "traccia.json": "application/json",
    "chitarra.mp3": "audio/mpeg",          # la chitarra isolata: quello che ascolta l'applicativo
}

app = FastAPI(title="Tablatore AI", docs_url="/api/docs")
_esecutore = ThreadPoolExecutor(max_workers=1)      # un brano alla volta (GPU condivisa)
_lavori: dict[str, dict] = {}
_blocco = threading.Lock()


# ---------------------------------------------------------------- utilità
def _nome_sicuro(nome: str) -> str:
    """Solo lettere, cifre, spazi, - e _ ; niente percorsi. Serve come titolo e nome del file."""
    stem = Path(nome or "brano").stem
    pulito = re.sub(r"[^\w\- ]+", "", stem, flags=re.UNICODE).strip()[:80]
    return pulito or "brano"


def _cartella(id_lavoro: str) -> Path:
    if not ID_VALIDO.match(id_lavoro):
        raise HTTPException(404, "Lavoro inesistente.")
    return LAVORI / id_lavoro


def _salva_stato(id_lavoro: str):
    with _blocco:
        stato = dict(_lavori[id_lavoro])
    (LAVORI / id_lavoro / "lavoro.json").write_text(json.dumps(stato, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")


def _aggiorna(id_lavoro: str, **campi):
    with _blocco:
        _lavori[id_lavoro].update(campi)


def _stato(id_lavoro: str) -> dict:
    """Dallo stato in memoria o, dopo un riavvio del server, da lavoro.json."""
    cartella = _cartella(id_lavoro)
    with _blocco:
        if id_lavoro in _lavori:
            return dict(_lavori[id_lavoro])
    f = cartella / "lavoro.json"
    if not f.exists():
        raise HTTPException(404, "Lavoro inesistente.")
    stato = json.loads(f.read_text(encoding="utf-8"))
    with _blocco:
        _lavori[id_lavoro] = stato
    return dict(stato)


def _per_pagina(r: RisultatoBrano, id_lavoro: str) -> dict:
    """Risultato senza percorsi del PC: al loro posto gli indirizzi di download."""
    d = asdict(r)
    for k in ("file_gp5", "file_tablatura", "file_testo", "file_midi", "file_traccia", "cartella", "file_note"):
        d.pop(k, None)
    d["file"] = {nome: f"/api/lavori/{id_lavoro}/file/{nome}" for nome in FILE_SCARICABILI
                 if (LAVORI / id_lavoro / nome).exists()}
    return d


def _esegui(id_lavoro: str, mp3: Path, accordatura: str, sensibilita: str, resoconto_llm: bool, titolo: str,
            stile: str = "auto", separatore: str = "demucs", ripetizioni: bool = False, semplifica: bool = False):
    _aggiorna(id_lavoro, stato="in_corso", inizio=time.time())
    try:
        r = trascrivi_brano(mp3, accordatura, sensibilita, cartella_output=mp3.parent, resoconto_llm=resoconto_llm,
                            su_passo=lambda p: _aggiorna(id_lavoro, passo=p), titolo=titolo, stile=stile,
                            separatore=separatore, ripetizioni=ripetizioni, semplifica=semplifica)
        (mp3.parent / "risultato.json").write_text(json.dumps(asdict(r), ensure_ascii=False), encoding="utf-8")
        _aggiorna(id_lavoro, stato="finito", passo=None, risultato=_per_pagina(r, id_lavoro))
    except Exception as e:                       # non dovrebbe succedere: l'orchestratore gestisce i guasti
        _aggiorna(id_lavoro, stato="errore", passo=None, errore=f"{type(e).__name__}: {e}")
    _salva_stato(id_lavoro)


# ---------------------------------------------------------------- API
@app.get("/api/opzioni")
def opzioni():
    return {"accordature": ["auto"] + list(config.ACCORDATURE),
            "corde": {k: v[0] for k, v in config.ACCORDATURE.items()},
            "sensibilita": ["auto"] + list(config.SENSIBILITA), "stili": list(STILI), "passi": PASSI,
            "separatori": list(config.SEPARATORI), "fattori_tempo": list(config.FATTORI_TEMPO),
            "dimensione_max_mb": config.DIMENSIONE_MAX_MB, "alphatab": config.ALPHATAB_VERSIONE,
            "banco_suoni": _banco_suoni(), "tutor": bool(os.getenv("OPENAI_API_KEY"))}


def _banco_suoni() -> str | None:
    """Banco di suoni locale dell'anteprima (GeneralUser GS, copia locale come chiede l'autore);
    None = la pagina usa il banco piccolo di alphaTab dal CDN."""
    f = STATICI / "suoni" / config.BANCO_SUONI
    return f"/static/suoni/{config.BANCO_SUONI}" if f.is_file() else None


@app.post("/api/lavori")
async def nuovo_lavoro(file: UploadFile = File(...), accordatura: str = Form("auto"),
                       sensibilita: str = Form("auto"), resoconto_llm: bool = Form(True),
                       stile: str = Form("auto"), separatore: str = Form("demucs"),
                       ripetizioni: bool = Form(False), semplifica: bool = Form(False)):
    if Path(file.filename or "").suffix.lower() not in config.ESTENSIONI_AMMESSE:
        raise HTTPException(400, "Sono ammessi solo file .mp3.")
    if separatore not in config.SEPARATORI:
        raise HTTPException(400, "Separatore non riconosciuto.")
    if accordatura != "auto" and accordatura not in config.ACCORDATURE:
        raise HTTPException(400, "Accordatura non riconosciuta.")
    if sensibilita != "auto" and sensibilita not in config.SENSIBILITA:
        raise HTTPException(400, "Sensibilità non riconosciuta.")
    if stile not in STILI:
        raise HTTPException(400, "Stile non riconosciuto.")
    id_lavoro = uuid.uuid4().hex
    cartella = LAVORI / id_lavoro
    cartella.mkdir(parents=True)
    titolo = _nome_sicuro(file.filename)
    mp3 = cartella / f"{titolo}.mp3"
    limite, scritti = config.DIMENSIONE_MAX_MB * 2**20, 0
    with open(mp3, "wb") as out:                  # a pezzi: il limite vale anche per file enormi
        while pezzo := await file.read(2**20):
            scritti += len(pezzo)
            if scritti > limite:
                out.close()
                mp3.unlink(missing_ok=True)
                cartella.rmdir()
                raise HTTPException(413, f"File troppo grande: il massimo è {config.DIMENSIONE_MAX_MB} MB.")
            out.write(pezzo)
    with _blocco:
        in_attesa = sum(1 for l in _lavori.values() if l["stato"] in ("in_coda", "in_corso"))
        _lavori[id_lavoro] = {"id": id_lavoro, "stato": "in_coda", "titolo": titolo, "passo": None,
                              "accordatura_richiesta": accordatura, "sensibilita_richiesta": sensibilita,
                              "creato": time.time(), "davanti_in_coda": in_attesa}
    _salva_stato(id_lavoro)
    _esecutore.submit(_esegui, id_lavoro, mp3, accordatura, sensibilita, resoconto_llm, titolo, stile, separatore,
                      ripetizioni, semplifica)
    return {"id": id_lavoro}


@app.get("/api/lavori/{id_lavoro}")
def stato_lavoro(id_lavoro: str):
    return _stato(id_lavoro)


@app.post("/api/lavori/{id_lavoro}/accordatura")
def cambia_accordatura(id_lavoro: str, accordatura: str = Form(...), stile: str | None = Form(None),
                       ripetizioni: bool | None = Form(None), semplifica: bool | None = Form(None),
                       tempo: float | None = Form(None)):
    stato = _stato(id_lavoro)
    if stato.get("stato") != "finito":
        raise HTTPException(409, "Il lavoro non è ancora finito.")
    f = _cartella(id_lavoro) / "risultato.json"
    if not f.exists():
        raise HTTPException(409, "Risultato non disponibile.")
    r = RisultatoBrano(**json.loads(f.read_text(encoding="utf-8")))
    try:
        nuovo = riaccorda(r, accordatura, stile=stile, ripetizioni=ripetizioni, semplifica=semplifica, tempo=tempo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    f.write_text(json.dumps(asdict(nuovo), ensure_ascii=False), encoding="utf-8")
    _aggiorna(id_lavoro, risultato=_per_pagina(nuovo, id_lavoro))
    _salva_stato(id_lavoro)
    return _stato(id_lavoro)


class Messaggio(BaseModel):
    ruolo: str = Field(pattern="^(utente|tutor)$")
    testo: str = Field(max_length=4000)


class DomandaTutor(BaseModel):
    domanda: str = Field(max_length=2000)          # il limite vero (TUTOR_DOMANDA_MAX) lo controlla il guardrail
    storia: list[Messaggio] = Field(default_factory=list, max_length=40)
    id_lavoro: str | None = None


_blocco_tutor = threading.Lock()                   # Qdrant locale: un client, una domanda alla volta


@app.post("/api/tutor")
def chiedi_al_tutor(d: DomandaTutor):
    """Tutor di teoria (RAG). Se c'è un lavoro finito, il tutor riceve i dati della sua tablatura.
    Alla prima domanda (o se le schede sono cambiate) l'indice viene creato: ~6 s, ~$0,0001."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "Tutor non disponibile: manca la chiave OPENAI_API_KEY nel file .env.")
    brano = None
    if d.id_lavoro:
        stato = _stato(d.id_lavoro)
        if stato.get("stato") == "finito":
            brano = stato.get("risultato")
    with _blocco_tutor:
        try:
            if not tutor.stato_indice()["pronto"]:
                tutor.indicizza()
            r = tutor.chiedi(d.domanda, [m.model_dump() for m in d.storia], brano)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return {"risposta": r.risposta, "fonti": r.fonti, "citate": r.citate, "da_llm": r.da_llm,
            "rifiutata": r.rifiutata, "scartata": r.scartata, "costo_usd": r.costo_usd, "tempo_s": r.tempo_s}


@app.get("/api/lavori/{id_lavoro}/file/{nome}")
def scarica(id_lavoro: str, nome: str):
    if nome not in FILE_SCARICABILI:
        raise HTTPException(404, "File non disponibile.")
    f = _cartella(id_lavoro) / nome
    if not f.is_file():
        raise HTTPException(404, "File non disponibile.")
    titolo = _stato(id_lavoro).get("titolo", "brano")
    return FileResponse(f, media_type=FILE_SCARICABILI[nome],
                        filename=f"{titolo} - {nome}".replace("_verificata", ""),
                        headers={"Cache-Control": "no-store"})


def _html(nome: str) -> str:
    """Pagina con la versione di alphaTab e un numero di versione dei file statici (data
    dell'ultima modifica): dopo un aggiornamento il browser non usa più i vecchi script dalla
    cache (successo nelle prove: Chrome mostrava la pagina con il vecchio app.js)."""
    versione = str(int(max(f.stat().st_mtime for f in STATICI.iterdir())))
    return ((STATICI / nome).read_text(encoding="utf-8")
            .replace("{{ALPHATAB}}", config.ALPHATAB_VERSIONE).replace("{{V}}", versione))


SENZA_CACHE = {"Cache-Control": "no-store"}     # la pagina punta sempre agli script dell'ultima versione


@app.get("/", response_class=HTMLResponse)
def pagina():
    return HTMLResponse(_html("index.html"), headers=SENZA_CACHE)


@app.get("/stampa", response_class=HTMLResponse)
def pagina_stampa():
    return HTMLResponse(_html("stampa.html"), headers=SENZA_CACHE)


app.mount("/static", StaticFiles(directory=STATICI), name="static")


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    LAVORI.mkdir(parents=True, exist_ok=True)
    print(f"Apri http://{config.HOST}:{config.PORTA} nel browser (Ctrl+C per fermare)")
    uvicorn.run(app, host=config.HOST, port=config.PORTA)
