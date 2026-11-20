"""Tool 1 - valida_file: guardrail in ingresso del Guardiano dell'input.

Controlla che il file caricato sia un MP3 vero e utilizzabile PRIMA di avviare
le elaborazioni costose (separazione, trascrizione). Non usa LLM: è codice
deterministico, veloce e gratuito.

Controlli bloccanti (il file viene rifiutato):
  - il file esiste e non è vuoto
  - estensione ammessa (.mp3)
  - dimensione entro il limite
  - ffprobe lo riconosce come audio MP3 (scopre i file rinominati, es. un .txt chiamato .mp3)
  - durata entro i limiti
  - abbastanza audio non silenzioso (file muto o quasi)

Avvisi non bloccanti (il file passa, ma l'utente viene informato):
  - silenzi lunghi all'inizio, alla fine o a metà (registrazioni manuali)
  - bitrate o frequenza di campionamento bassi
  - avvisi fissi: copyright e trascrizione approssimativa
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datapizza.tools import tool

from app import config


@dataclass
class RisultatoValidazione:
    valido: bool
    errori: list[str] = field(default_factory=list)
    avvisi: list[str] = field(default_factory=list)
    metadati: dict = field(default_factory=dict)


def _ffprobe(percorso: Path) -> dict | None:
    """Legge formato e stream audio con ffprobe. Ritorna None se il file non è leggibile."""
    exe = shutil.which("ffprobe")
    if exe is None:
        raise RuntimeError("ffprobe non trovato: installare ffmpeg e aggiungerlo al PATH")
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(percorso)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def _silenzi(percorso: Path, durata: float) -> list[tuple[float, float]]:
    """Tratti di silenzio lunghi (filtro silencedetect di ffmpeg): [(inizio, fine), ...]."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg non trovato: installare ffmpeg e aggiungerlo al PATH")
    filtro = f"silencedetect=noise={config.SOGLIA_SILENZIO_DB}dB:d={config.SILENZIO_MIN_S}"
    proc = subprocess.run([exe, "-hide_banner", "-nostats", "-i", str(percorso), "-af", filtro,
                           "-f", "null", "-"], capture_output=True, text=True, timeout=120)
    tratti, inizio = [], None
    for riga in proc.stderr.splitlines():
        if "silence_start:" in riga:
            inizio = max(0.0, float(riga.split("silence_start:")[1].split()[0]))
        elif "silence_end:" in riga and inizio is not None:
            tratti.append((inizio, float(riga.split("silence_end:")[1].split()[0])))
            inizio = None
    if inizio is not None:                      # silenzio che arriva fino alla fine del file
        tratti.append((inizio, durata))
    return tratti


def valida_file(percorso: str | Path) -> RisultatoValidazione:
    """Valida un file audio caricato dall'utente. Funzione pura, usabile anche senza agente."""
    p = Path(percorso)
    errori: list[str] = []
    avvisi: list[str] = []
    meta: dict = {"nome": p.name}

    # 1. Esistenza e dimensione
    if not p.is_file():
        return RisultatoValidazione(False, [f"File non trovato: {p.name}"])
    dim = p.stat().st_size
    meta["dimensione_mb"] = round(dim / 1_048_576, 2)
    if dim == 0:
        return RisultatoValidazione(False, ["Il file è vuoto."], metadati=meta)
    if dim > config.DIMENSIONE_MAX_MB * 1_048_576:
        errori.append(f"File troppo grande: {meta['dimensione_mb']} MB (massimo {config.DIMENSIONE_MAX_MB} MB).")

    # 2. Estensione
    if p.suffix.lower() not in config.ESTENSIONI_AMMESSE:
        errori.append(f"Formato non ammesso: '{p.suffix or 'nessuna estensione'}'. Carica un file MP3.")

    # 3. Contenuto reale (non ci si fida dell'estensione)
    info = _ffprobe(p)
    stream = None
    if info:
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if stream is None:
        errori.append("Il file non contiene audio leggibile (file danneggiato o non audio).")
        return RisultatoValidazione(False, errori, metadati=meta)

    codec = stream.get("codec_name")
    meta["codec"] = codec
    if p.suffix.lower() == ".mp3" and codec != "mp3":
        errori.append(f"Il file ha estensione .mp3 ma contiene audio '{codec}'. Carica un MP3 vero.")

    durata = float(info["format"].get("duration") or stream.get("duration") or 0)
    sample_rate = int(stream.get("sample_rate") or 0)
    bitrate = int(info["format"].get("bit_rate") or stream.get("bit_rate") or 0)
    meta.update(durata_s=round(durata, 2), sample_rate=sample_rate,
                canali=stream.get("channels"), bitrate_kbps=bitrate // 1000)

    # 4. Durata
    if durata < config.DURATA_MIN_S:
        errori.append(f"Brano troppo corto: {durata:.1f} s (minimo {config.DURATA_MIN_S:.0f} s).")
    elif durata > config.DURATA_MAX_S:
        errori.append(f"Brano troppo lungo: {durata / 60:.1f} min (massimo {config.DURATA_MAX_S / 60:.0f} min).")

    # 5. Silenzi (registrazioni manuali: momenti morti all'inizio, alla fine o in mezzo)
    if not errori:
        tratti = _silenzi(p, durata)
        iniziale = next((round(b, 2) for a, b in tratti if a <= 0.05), 0.0)
        finale = next((round(durata - a, 2) for a, b in tratti if b >= durata - 0.05 and a > 0.05), 0.0)
        totale = sum(b - a for a, b in tratti)
        meta.update(silenzio_iniziale_s=iniziale, silenzio_finale_s=finale,
                    durata_musica_s=round(durata - totale, 2))
        if durata - totale < config.DURATA_MIN_S:
            errori.append(f"Il file è quasi tutto silenzioso: solo {durata - totale:.1f} s di audio "
                          f"(minimo {config.DURATA_MIN_S:.0f} s).")
        else:
            if iniziale:
                avvisi.append(f"{iniziale:.1f} s di silenzio all'inizio: la tablatura partirà dall'inizio della musica.")
            if finale:
                avvisi.append(f"{finale:.1f} s di silenzio alla fine: verranno ignorati.")
            interni = [(a, b) for a, b in tratti if a > 0.05 and b < durata - 0.05]
            if interni:
                elenco = ", ".join(f"{a:.1f}–{b:.1f} s" for a, b in interni)
                avvisi.append(f"Pause di silenzio a metà brano ({elenco}): se è una registrazione con "
                              "interruzioni, il tempo e le battute potrebbero sfasarsi.")

    # 6. Qualità (solo avvisi)
    if 0 < bitrate < config.BITRATE_MIN_CONSIGLIATO:
        avvisi.append(f"Bitrate basso ({bitrate // 1000} kbps): la trascrizione potrebbe essere meno precisa.")
    if 0 < sample_rate < config.SAMPLE_RATE_MIN_CONSIGLIATO:
        avvisi.append(f"Frequenza di campionamento bassa ({sample_rate} Hz): la trascrizione potrebbe essere meno precisa.")

    # 7. Avvisi fissi, sempre presenti
    avvisi += [config.AVVISO_APPROSSIMAZIONE, config.AVVISO_COPYRIGHT]

    return RisultatoValidazione(not errori, errori, avvisi, meta)


@tool
def valida_file_audio(percorso: str) -> str:
    """Controlla che il file audio caricato dall'utente sia un MP3 valido (formato, dimensione,
    durata, contenuto audio reale) prima di avviare la trascrizione. Restituisce un JSON con:
    'valido' (true/false), 'errori' (motivi del rifiuto), 'avvisi' (da mostrare all'utente),
    'metadati' (durata, codec, bitrate...). Se 'valido' è false la trascrizione NON va avviata."""
    return json.dumps(asdict(valida_file(percorso)), ensure_ascii=False)
