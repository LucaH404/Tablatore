"""Test del server MCP (Step 26). Costo 0: niente trascrizioni né LLM; il protocollo si prova davvero
avviando il server via stdio con il client MCP di datapizza-ai."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from app import mcp_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trascrizione(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "CARTELLA_MCP", tmp_path)
    d = tmp_path / "Brano di prova"
    d.mkdir()
    (d / "risultato.json").write_text(json.dumps({"completato": True, "modalita": "codice", "titolo": "Brano di prova",
                                                  "accordatura": "standard", "bpm": 100.0, "numero_battute": 3}),
                                      encoding="utf-8")
    battute = [{"numero": n, "slot_battuta": 16,
                "eventi": [{"slot": 0, "durata_slot": 16, "note": [{"corda": 6, "tasto": n + 2}]}]} for n in (1, 2, 3)]
    (d / "tablatura_verificata.json").write_text(json.dumps({"titolo": "Brano di prova", "accordatura": "standard",
                                                             "bpm": 100.0, "battute": battute}), encoding="utf-8")
    return "Brano di prova"


def test_id_e_percorsi_controllati(trascrizione):
    for cattivo in ("../../etc", "a/b", "", "x" * 200):
        with pytest.raises(ValueError):
            mcp_server.leggi_tablatura(cattivo)
    with pytest.raises(ValueError, match="inesistente"):
        mcp_server.leggi_tablatura("non esiste")
    with pytest.raises(ValueError, match=r"\.mp3"):
        mcp_server.trascrivi_brano(str(ROOT / "README.md"))
    with pytest.raises(ValueError, match=r"\.mp3"):
        mcp_server.trascrivi_brano("C:/non/esiste.mp3")


def test_leggi_tablatura_a_intervalli(trascrizione):
    tutto = mcp_server.leggi_tablatura(trascrizione)
    assert "battute 1-3 di 3" in tutto and "accordatura standard" in tutto
    pezzo = mcp_server.leggi_tablatura(trascrizione, da_battuta=2, a_battuta=2)
    assert "battute 2-2 di 3" in pezzo and "4--" in pezzo and "3--" not in pezzo     # battuta 2 = tasto 4
    assert "Nessuna battuta" in mcp_server.leggi_tablatura(trascrizione, da_battuta=10)


def test_elenca_trascrizioni(trascrizione):
    assert mcp_server.elenca_trascrizioni() == [{"id_trascrizione": "Brano di prova", "titolo": "Brano di prova",
                                                 "accordatura": "standard", "bpm": 100.0, "battute": 3}]


def test_protocollo_stdio_con_client_datapizza():
    """Il server parte come processo separato e risponde sul protocollo MCP (stdout pulito)."""
    from datapizza.tools.mcp_client import MCPClient

    async def prova():
        async with MCPClient(command=sys.executable, args=["-m", "app.mcp_server"], timeout=60,
                             env={"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}) as client:
            nomi = [t.name for t in await client.a_list_tools()]
            errore = await client.call_tool("leggi_tablatura", {"id_trascrizione": "../../etc"})
            return nomi, errore

    nomi, errore = asyncio.run(prova())
    assert nomi == ["trascrivi_brano", "rifai_tablatura", "leggi_tablatura", "chiedi_al_tutor", "elenca_trascrizioni"]
    assert errore.isError and "non valido" in errore.content[0].text
