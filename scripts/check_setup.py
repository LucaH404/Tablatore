"""Verifica dell'ambiente (Fase 0, ampliata nello Step 27 per lo script di inizializzazione).

Controlla: Python 3.11; ffmpeg/ffprobe; i pacchetti principali e le versioni che contano (mcp 1.x per il
client MCP di datapizza, O77); PyTorch e GPU; modelli già in cache; banco di suoni; chiave OpenAI e una
chiamata LLM via datapizza-ai (≈ $0,00001; con --senza-llm si salta).
I controlli facoltativi (GPU, banco di suoni, modelli in cache, chiave) non rendono l'ambiente "non pronto".
Uso:  .venv\\Scripts\\python.exe scripts\\check_setup.py [--senza-llm]
"""
import importlib
import importlib.metadata as md
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

ok = True


def check(nome, esito, dettaglio="", facoltativo=False):
    global ok
    if not facoltativo:
        ok &= bool(esito)
    stato = "OK" if esito else ("--" if facoltativo else "KO")
    print(f"[{stato}] {nome}" + (f" - {dettaglio}" if dettaglio else ""))


check("Python 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0])
for exe in ("ffmpeg", "ffprobe"):
    check(exe, shutil.which(exe), shutil.which(exe) or "non trovato nel PATH")

for modulo, pacchetto in [("torch", "torch"), ("torchaudio", "torchaudio"), ("demucs", "demucs"),
                          ("basic_pitch", "basic-pitch"), ("beat_this", "beat-this"), ("librosa", "librosa"),
                          ("onnxruntime", "onnxruntime"), ("guitarpro", "pyguitarpro"), ("datapizza", "datapizza-ai"),
                          ("qdrant_client", "qdrant-client"), ("fastapi", "fastapi"), ("mcp", "mcp")]:
    try:
        importlib.import_module(modulo)
        check(f"pacchetto {pacchetto}", True, md.version(pacchetto))
    except Exception as e:  # noqa: BLE001
        check(f"pacchetto {pacchetto}", False, f"{type(e).__name__}: {str(e)[:120]}")

try:
    v = md.version("mcp")
    check("mcp 1.x (client MCP di datapizza)", v.startswith("1."), v + ("" if v.startswith("1.") else " -> serve mcp==1.30.0"))
    from datapizza.tools.mcp_client import MCPClient  # noqa: F401
except Exception as e:  # noqa: BLE001
    check("client MCP di datapizza", False, str(e)[:120])

try:
    import torch
    gpu = torch.cuda.is_available()
    check("GPU CUDA", gpu, torch.cuda.get_device_name(0) if gpu else "non disponibile: si usa la CPU (più lento)",
          facoltativo=True)
except Exception:  # noqa: BLE001
    pass

cache_hub = Path(os.getenv("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
check("checkpoint beat_this in cache", any(cache_hub.glob("*final0*")) if cache_hub.exists() else False,
      str(cache_hub), facoltativo=True)
from app import config  # noqa: E402

sf2 = ROOT / "app" / "web" / "static" / "suoni" / config.BANCO_SUONI
check("banco di suoni GeneralUser GS", sf2.is_file(), "presente" if sf2.is_file() else "assente: si userà quello piccolo",
      facoltativo=True)

key = os.getenv("OPENAI_API_KEY")
check("OPENAI_API_KEY nel .env", key, "presente" if key else "assente: niente resoconto IA né tutor", facoltativo=True)

if key and "--senza-llm" not in sys.argv:
    from datapizza.clients.openai import OpenAIClient

    client = OpenAIClient(api_key=key, model=config.MODELLO_LLM)
    t0 = time.perf_counter()
    try:
        r = client.invoke("Rispondi solo con la parola: pronto")
        dt = time.perf_counter() - t0
        u = r.usage
        check("Chiamata LLM via datapizza-ai", r.text.strip(),
              f"risposta='{r.text.strip()}' | {dt:.2f}s | token in={u.prompt_tokens} out={u.completion_tokens}")
    except Exception as e:  # noqa: BLE001
        check("Chiamata LLM via datapizza-ai", False, str(e)[:200])
    try:
        from app import tutor
        s = tutor.stato_indice()
        check("indice del tutor", s["pronto"], f"{s.get('chunk', 0)} chunk" if s["pronto"] else
              "da creare (lo fa il server alla prima domanda)", facoltativo=True)
    except Exception as e:  # noqa: BLE001
        check("indice del tutor", False, str(e)[:120], facoltativo=True)

print("\nAMBIENTE PRONTO" if ok else "\nAMBIENTE NON PRONTO")
sys.exit(0 if ok else 1)
