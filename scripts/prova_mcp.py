"""Collaudo del server MCP (Step 26) con il client MCP di datapizza-ai, come lo userebbe un assistente.

Avvia app.mcp_server come processo separato (stdio), elenca i tool e li chiama. Con --agente un agente
datapizza (gpt-5.4-nano) riceve i tool MCP e risponde da solo a una richiesta in linguaggio naturale.

Uso: .venv\\Scripts\\python.exe scripts\\prova_mcp.py <file.mp3> [--agente]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from datapizza.tools.mcp_client import MCPClient  # noqa: E402


def _testo(risultato) -> str:
    blocchi = getattr(risultato, "content", None) or []
    return "\n".join(getattr(b, "text", str(b)) for b in blocchi)


async def main(mp3: str, agente: bool):
    server = MCPClient(command=sys.executable, args=["-m", "app.mcp_server"], timeout=300,
                       env={"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", **_env()})
    async with server as client:
        tools = await client.a_list_tools()
        print("TOOL ESPOSTI:", [t.name for t in tools])
        t0 = time.perf_counter()
        r = await client.call_tool("trascrivi_brano", {"percorso_mp3": mp3})
        riepilogo = json.loads(_testo(r))
        print(f"\ntrascrivi_brano ({time.perf_counter() - t0:.1f} s):",
              {k: riepilogo[k] for k in ("id_trascrizione", "accordatura", "bpm", "tonalita", "battute", "note")})
        tab = _testo(await client.call_tool("leggi_tablatura", {"id_trascrizione": riepilogo["id_trascrizione"],
                                                                 "da_battuta": 1, "a_battuta": 4}))
        print("\nleggi_tablatura (battute 1-4):\n" + tab)
        print("elenca_trascrizioni:", _testo(await client.call_tool("elenca_trascrizioni", {}))[:300])
        errore = _testo(await client.call_tool("leggi_tablatura", {"id_trascrizione": "../../etc"}))
        print("id non valido ->", errore[:120])
        if agente:
            from datapizza.agents import Agent
            from datapizza.clients.openai import OpenAIClient
            import os
            from app import config
            a = Agent(name="assistente", client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=config.MODELLO_LLM),
                      system_prompt="Aiuti un chitarrista usando i tool del Tablatore. Rispondi in italiano, breve.",
                      tools=tools, max_steps=6)
            t0 = time.perf_counter()
            risposta = await a.a_run(
                f"Nella trascrizione '{riepilogo['id_trascrizione']}' che accordatura e che tonalità ha il brano? "
                "Chiedi al tutor che scala usare per improvvisare su questo brano e riassumimi la risposta.")
            print(f"\nAGENTE datapizza con i tool MCP ({time.perf_counter() - t0:.1f} s):\n{risposta.text}")


def _env():
    import os
    return {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "USERPROFILE", "TEMP", "TMP",
                                                         "APPDATA", "LOCALAPPDATA", "OPENAI_API_KEY", "HOME")}


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], "--agente" in sys.argv))
