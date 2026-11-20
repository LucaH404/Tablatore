"""Server MCP del Tablatore AI (Step 26): i tool della pipeline usabili da un assistente compatibile
(Claude Desktop, Claude Code, un agente datapizza-ai con MCPClient...).

Avvio (trasporto stdio, come lo lancia un client MCP):
    .venv\\Scripts\\python.exe -m app.mcp_server

Tool esposti:
  - trascrivi_brano(percorso_mp3, accordatura, stile, sensibilita) -> riepilogo + id della trascrizione
  - rifai_tablatura(id_trascrizione, accordatura, stile, tempo, semplifica, ripetizioni)
  - leggi_tablatura(id_trascrizione, da_battuta, a_battuta) -> tablatura in testo
  - chiedi_al_tutor(domanda, id_trascrizione) -> risposta del tutor di teoria (RAG) con le fonti
  - elenca_trascrizioni()

Le trascrizioni vanno in output/mcp/<id>/ (stessa struttura della pagina web). Il server gira solo su
questo PC (stdio, nessuna porta aperta) e legge solo file .mp3 esistenti indicati dal client.

stdout è riservato al protocollo MCP: qualunque print della pipeline (riepilogo del tracing, librerie,
ffmpeg) lo corromperebbe. All'avvio il descrittore 1 viene copiato per il protocollo e poi rediretto su
stderr (anche per i sottoprocessi); dentro ogni tool sys.stdout punta a stderr.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARTELLA_MCP = ROOT / "output" / "mcp"
ID_VALIDO = re.compile(r"^[\w\- ]{1,80}$")
MAX_CARATTERI_TAB = 20000            # una tab intera di 4 minuti in testo è ~60 000 caratteri: si legge a pezzi


def _protocollo_su_stdio_pulito():
    """stdin e stdout sono il canale del protocollo MCP: si copiano per il protocollo, poi il descrittore 1
    va su stderr e lo 0 su un ingresso vuoto. Senza il secondo passo ffmpeg (validazione, separazione)
    ereditava lo stdin del protocollo e restava bloccato: trascrivi_brano arrivava al timeout (300 s)."""
    import io
    fd_out, fd_in = os.dup(1), os.dup(0)
    os.dup2(2, 1)                                     # sottoprocessi e librerie C: uscita su stderr
    nullo = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nullo, 0)                                 # sottoprocessi: ingresso vuoto
    os.close(nullo)
    sys.stdout = io.TextIOWrapper(os.fdopen(fd_out, "wb", buffering=0), encoding="utf-8", write_through=True)
    sys.stdin = io.TextIOWrapper(os.fdopen(fd_in, "rb", buffering=0), encoding="utf-8")


@contextlib.contextmanager
def _silenzio():
    """Durante un tool i print di Python vanno su stderr (lo stdout del protocollo resta intatto)."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _cartella(id_trascrizione: str) -> Path:
    if not ID_VALIDO.match(id_trascrizione or ""):
        raise ValueError("id_trascrizione non valido: usa quello restituito da trascrivi_brano o elenca_trascrizioni.")
    d = CARTELLA_MCP / id_trascrizione
    if not (d / "risultato.json").exists():
        raise ValueError(f"Trascrizione '{id_trascrizione}' inesistente.")
    return d


def _riepilogo(r, id_trascrizione: str) -> dict:
    from app.orchestratore import AVVISI_FISSI
    return {"id_trascrizione": id_trascrizione, "completato": r.completato, "motivo_stop": r.motivo_stop,
            "titolo": r.titolo, "accordatura": r.accordatura, "bpm": r.bpm, "metro": r.metro,
            "tonalita": r.tonalita, "battute": r.numero_battute, "note": r.numero_note,
            "note_tolte_dal_verificatore": r.note_tolte_verifica, "suonabile": r.suonabile,
            "decisioni": r.decisioni, "avvisi": [a for a in r.avvisi if a not in AVVISI_FISSI],
            "avvisi_fissi": AVVISI_FISSI, "tempo_s": r.tempo_totale_s,
            "file": {k: v for k, v in (("guitar_pro", r.file_gp5), ("testo", r.file_testo),
                                       ("midi", r.file_midi)) if v},
            "suggerimento": "Usa leggi_tablatura per vedere le battute e chiedi_al_tutor per domande sul brano."}


def _carica(id_trascrizione: str):
    from app.orchestratore import RisultatoBrano
    d = _cartella(id_trascrizione)
    return RisultatoBrano(**json.loads((d / "risultato.json").read_text(encoding="utf-8"))), d


# ---------------------------------------------------------------- funzioni dei tool (testabili senza MCP)
def trascrivi_brano(percorso_mp3: str, accordatura: str = "auto", stile: str = "auto",
                    sensibilita: str = "auto") -> dict:
    from app.orchestratore import trascrivi_brano as pipeline
    p = Path(percorso_mp3).expanduser()
    if p.suffix.lower() != ".mp3" or not p.is_file():
        raise ValueError(f"Serve il percorso di un file .mp3 esistente: {percorso_mp3}")
    id_trascrizione = re.sub(r"[^\w\- ]+", "", p.stem).strip()[:60] or "brano"
    out = CARTELLA_MCP / id_trascrizione
    with _silenzio():
        r = pipeline(p, accordatura, sensibilita, cartella_output=out, resoconto_llm=False, stile=stile,
                     titolo=p.stem)
    out.mkdir(parents=True, exist_ok=True)
    (out / "risultato.json").write_text(json.dumps(asdict(r), ensure_ascii=False), encoding="utf-8")
    return _riepilogo(r, id_trascrizione)


def rifai_tablatura(id_trascrizione: str, accordatura: str = "auto", stile: str | None = None,
                    tempo: float | None = None, semplifica: bool | None = None,
                    ripetizioni: bool | None = None) -> dict:
    from app.orchestratore import riaccorda
    r, d = _carica(id_trascrizione)
    with _silenzio():
        nuovo = riaccorda(r, accordatura, stile=stile, tempo=tempo, semplifica=semplifica, ripetizioni=ripetizioni)
    (d / "risultato.json").write_text(json.dumps(asdict(nuovo), ensure_ascii=False), encoding="utf-8")
    return _riepilogo(nuovo, id_trascrizione)


def leggi_tablatura(id_trascrizione: str, da_battuta: int = 1, a_battuta: int | None = None) -> str:
    from app import config
    from app.tools.tablatura import _testo
    _, d = _carica(id_trascrizione)
    tab = json.loads((d / "tablatura_verificata.json").read_text(encoding="utf-8"))
    battute = [b for b in tab["battute"] if b["numero"] >= da_battuta and (a_battuta is None or b["numero"] <= a_battuta)]
    if not battute:
        return f"Nessuna battuta nell'intervallo richiesto (il brano ha {len(tab['battute'])} battute)."
    corde = config.ACCORDATURE[tab["accordatura"]][0]
    testo = (f"{tab.get('titolo', '')} | accordatura {tab['accordatura']} | {tab['bpm']} BPM | battute "
             f"{battute[0]['numero']}-{battute[-1]['numero']} di {len(tab['battute'])}\n"
             "(riga in alto = 1ª corda, la più acuta; numeri = tasti; 3 caratteri per semicroma)\n\n"
             + _testo(battute, corde))
    if len(testo) > MAX_CARATTERI_TAB:
        taglio = testo[:MAX_CARATTERI_TAB].rsplit("\n\n", 1)[0]
        testo = taglio + "\n\n[...] Testo troppo lungo: chiedi un intervallo di battute più piccolo."
    return testo


def chiedi_al_tutor(domanda: str, id_trascrizione: str | None = None) -> dict:
    from app import tutor
    from app.orchestratore import _dati_resoconto
    brano = None
    if id_trascrizione:
        r, _ = _carica(id_trascrizione)
        brano = {**_dati_resoconto(r), "titolo": r.titolo, "numero_battute": r.numero_battute,
                 "numero_note": r.numero_note, "stile": r.stile, "decisioni": r.decisioni}
    with _silenzio():
        if not tutor.stato_indice()["pronto"]:
            tutor.indicizza()
        risposta = tutor.chiedi(domanda, brano=brano)
    return {"risposta": risposta.risposta, "fonti": risposta.citate or risposta.fonti,
            "rifiutata": risposta.rifiutata, "costo_usd": risposta.costo_usd}


def elenca_trascrizioni() -> list[dict]:
    out = []
    for d in sorted(CARTELLA_MCP.glob("*/risultato.json")):
        r = json.loads(d.read_text(encoding="utf-8"))
        out.append({"id_trascrizione": d.parent.name, "titolo": r.get("titolo"), "accordatura": r.get("accordatura"),
                    "bpm": r.get("bpm"), "battute": r.get("numero_battute")})
    return out


# ---------------------------------------------------------------- server MCP
def crea_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("tablatore-ai", instructions=(
        "Tool per trasformare un MP3 in tablatura per chitarra e per studiarla. Flusso tipico: trascrivi_brano "
        "con il percorso di un .mp3 (10-60 s) -> leggi_tablatura a pezzi (es. 8 battute alla volta) -> "
        "chiedi_al_tutor per teoria e tecnica. Se l'accordatura o il tempo sembrano sbagliati usa "
        "rifai_tablatura. La trascrizione è automatica e approssimativa; le tab di brani protetti sono solo "
        "per uso personale e di studio."))

    @mcp.tool(name="trascrivi_brano")
    def trascrivi_brano_mcp(percorso_mp3: str, accordatura: str = "auto", stile: str = "auto",
                            sensibilita: str = "auto") -> dict:
        """Trascrive un file MP3 in tablatura (separazione della chitarra, ritmo, note, diteggiatura, verifica).
        Richiede 10-60 secondi. 'accordatura': "auto" oppure standard, drop D, Eb standard, D standard, drop C#,
        drop C, C# standard, C standard, drop B. 'stile': auto, power_chord, libero. 'sensibilita': auto, bassa,
        media, alta. Restituisce accordatura, BPM, tonalità, battute, avvisi, decisioni e l'id_trascrizione."""
        return trascrivi_brano(percorso_mp3, accordatura, stile, sensibilita)

    @mcp.tool(name="rifai_tablatura")
    def rifai_tablatura_mcp(id_trascrizione: str, accordatura: str = "auto", stile: str | None = None,
                            tempo: float | None = None, semplifica: bool | None = None,
                            ripetizioni: bool | None = None) -> dict:
        """Rifà solo la tablatura di una trascrizione esistente (0,1-5 s) cambiando accordatura, stile, tempo
        (0.5 = metà, 1 = rilevato, 2 = doppio), semplificazione o confronto tra ripetizioni."""
        return rifai_tablatura(id_trascrizione, accordatura, stile, tempo, semplifica, ripetizioni)

    @mcp.tool(name="leggi_tablatura")
    def leggi_tablatura_mcp(id_trascrizione: str, da_battuta: int = 1, a_battuta: int | None = None) -> str:
        """Restituisce la tablatura in testo (riga in alto = corda più acuta), dalla battuta 'da_battuta' alla
        'a_battuta' compresa. Per brani lunghi chiedi pochi blocchi alla volta (es. 8 battute)."""
        return leggi_tablatura(id_trascrizione, da_battuta, a_battuta)

    @mcp.tool(name="chiedi_al_tutor")
    def chiedi_al_tutor_mcp(domanda: str, id_trascrizione: str | None = None) -> dict:
        """Domanda al tutor di teoria della chitarra (RAG su schede di teoria, tecnica e uso del Tablatore).
        Con 'id_trascrizione' il tutor conosce accordatura, tonalità e decisioni di quel brano (es. "che scala
        uso per improvvisare?"). Restituisce la risposta con le schede citate. Costo ~$0,0006 a domanda."""
        return chiedi_al_tutor(domanda, id_trascrizione)

    @mcp.tool(name="elenca_trascrizioni")
    def elenca_trascrizioni_mcp() -> list[dict]:
        """Elenca le trascrizioni già fatte con questo server (id, titolo, accordatura, BPM, battute)."""
        return elenca_trascrizioni()

    return mcp


def main():
    _protocollo_su_stdio_pulito()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    CARTELLA_MCP.mkdir(parents=True, exist_ok=True)
    try:
        crea_server().run("stdio")
    finally:
        with contextlib.suppress(Exception):
            from app import tutor
            tutor.chiudi_indice()


if __name__ == "__main__":
    main()
