"""Inizializzazione, passi comuni a Windows/macOS/Linux (chiamato da inizializza.ps1 / inizializza.sh).

Scarica in anticipo ciò che altrimenti arriverebbe al primo uso:
  - modello Demucs htdemucs_6s (separazione) e checkpoint beat_this "final0" (ritmo, 77 MB);
  - banco di suoni GeneralUser GS dell'anteprima (31 MB, copia locale come chiede l'autore);
  - (con --sperimentali) pesi di BS-RoFormer (700 MB);
  - indice del tutor (solo se c'è OPENAI_API_KEY nel .env: ~6 s, ~$0,0001).
Ogni passo è indipendente: se uno fallisce si continua e alla fine si stampa il riepilogo.

Uso: python scripts/prepara_risorse.py [--sperimentali] [--senza-modelli] [--senza-suoni] [--senza-tutor]
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL_SUONI = "https://cdn.jsdelivr.net/npm/generaluser@1.47.1/GeneralUser.sf2"
BYTE_SUONI = 31_281_186
esiti: list[tuple[str, bool, str]] = []


def passo(nome: str):
    def deco(f):
        def eseguito(*a, **k):
            print(f"\n==> {nome}", flush=True)
            t0 = time.perf_counter()
            try:
                dettaglio = f(*a, **k) or "fatto"
                esiti.append((nome, True, f"{dettaglio} ({time.perf_counter() - t0:.0f} s)"))
            except Exception as e:  # noqa: BLE001 - un passo fallito non ferma gli altri
                esiti.append((nome, False, f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"    {'ok' if esiti[-1][1] else 'ERRORE'}: {esiti[-1][2]}", flush=True)
        return eseguito
    return deco


@passo("Modello Demucs htdemucs_6s (separazione della chitarra)")
def demucs():
    from app import config
    from demucs.api import Separator
    Separator(config.MODELLO_DEMUCS)
    return "in cache"


@passo("Checkpoint beat_this 'final0' (ritmo)")
def beat_this():
    from app import config
    from beat_this.inference import Audio2Beats
    Audio2Beats(checkpoint_path=config.MODELLO_BEAT, device="cpu", dbn=False)
    return "in cache"


@passo("Pesi BS-RoFormer (separatore sperimentale, 700 MB)")
def roformer():
    from app import config
    from bs_roformer import ensure_model_assets      # se manca bs-roformer-infer: ImportError chiaro
    ensure_model_assets(config.MODELLO_ROFORMER)
    return "in cache"


@passo("Banco di suoni GeneralUser GS (anteprima, 31 MB)")
def suoni():
    from app import config
    dest = ROOT / "app" / "web" / "static" / "suoni" / config.BANCO_SUONI
    if dest.is_file() and dest.stat().st_size == BYTE_SUONI:
        return "già presente"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parziale")
    urllib.request.urlretrieve(URL_SUONI, tmp)
    if tmp.stat().st_size != BYTE_SUONI or tmp.read_bytes()[:4] != b"RIFF":
        tmp.unlink()
        raise RuntimeError("file scaricato diverso da quello atteso (dimensione o intestazione)")
    tmp.replace(dest)
    return f"scaricato in {dest.relative_to(ROOT)}"


@passo("Indice del tutor di teoria (RAG)")
def tutor():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        return "saltato: nessuna OPENAI_API_KEY nel .env (lo crea il server alla prima domanda)"
    from app import tutor as t
    if t.stato_indice()["pronto"]:
        return "già aggiornato"
    d = t.indicizza()
    return f"{d['schede']} schede, {d['chunk']} chunk, ≈ ${d['costo_stimato_usd']}"


def main():
    args = set(sys.argv[1:])
    if "--senza-modelli" not in args:
        demucs()
        beat_this()
        if "--sperimentali" in args:
            roformer()
    if "--senza-suoni" not in args:
        suoni()
    if "--senza-tutor" not in args:
        tutor()
    print("\nRIEPILOGO RISORSE")
    for nome, ok, dettaglio in esiti:
        print(f"  [{'OK' if ok else 'KO'}] {nome}: {dettaglio}")
    sys.exit(0 if all(ok for _, ok, _ in esiti) else 1)


if __name__ == "__main__":
    main()
