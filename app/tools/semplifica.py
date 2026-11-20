"""Tool 4d - semplifica_note: un accordo riattaccato mentre sta ancora suonando diventa uno solo.

Problema (Step 19, segnalato dall'utente su "Lying From You"): sulla chitarra distorta il suono
tenuto "pulsa" e Basic Pitch legge una nota lunga come più attacchi ravvicinati; il passo dei
power chord trasforma ogni frammento in un accordo intero -> "2-2-2" dove l'originale ha un colpo.
Misurato con la metrica D (attacchi abbinati uno a uno con Songsterr):
  - Lying From You (3 riff, tempo 80): attacchi scritti 43 -> 32, precisione 63% -> 75%,
    richiamo 90% -> 80%;
  - Nine Thou: precisione sale, richiamo scende, F1 degli attacchi invariato (~55%), tasti giusti
    per beat -2 punti: i riattacchi veri dello stesso accordo (es. "6-6") a volte si perdono.
Nessun indizio dell'audio separa i frammenti dai colpi veri (forza dell'attacco, calo di volume,
posizione sulla griglia: AUC 0,39-0,64). Per questo è un'opzione ("Semplifica le note ripetute"),
spenta di default: tablatura più pulita a vista, a costo di qualche colpo vero.

Regola: un gruppo di note con le STESSE altezze del gruppo precedente, attaccato mentre il
precedente suona ancora e a meno di SOGLIA_UNISCI_BEAT beat di distanza, viene unito al
precedente (che si allunga fino alla fine del frammento).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config
from app.tools.power_chord import _gruppi


@dataclass
class RisultatoSemplifica:
    file_note: str
    accordi_prima: int
    accordi_uniti: int
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)


def unisci_note_tenute(note: list[dict], beat: list[float], soglia_beat: float = config.SOGLIA_UNISCI_BEAT
                       ) -> tuple[list[dict], int, int]:
    """Note con i riattacchi uniti, numero di accordi uniti e numero di accordi iniziali."""
    gruppi = _gruppi([dict(n) for n in note])
    if not gruppi:
        return [], 0, 0
    b = np.asarray(beat, dtype=float)
    tenuti, uniti = [gruppi[0]], 0
    for g in gruppi[1:]:
        p = tenuti[-1]
        t0, t1 = p[0]["inizio"], g[0]["inizio"]
        if len(b) >= 2:
            k = int(np.clip(np.searchsorted(b, t1) - 1, 0, len(b) - 2))
            durata_beat = b[k + 1] - b[k]
        else:
            durata_beat = 0.5
        stesse = {n["midi"] for n in g} == {n["midi"] for n in p}
        suona = max(n["fine"] for n in p) >= t1 - 0.03
        if stesse and suona and (t1 - t0) / durata_beat < soglia_beat:
            fine = max(n["fine"] for n in g)
            for n in p:
                n["fine"] = max(n["fine"], fine)
            uniti += 1
        else:
            tenuti.append(g)
    return [n for g in tenuti for n in g], uniti, len(gruppi)


def semplifica_note(file_note: str | Path, file_ritmo: str | Path,
                    cartella_output: str | Path | None = None) -> RisultatoSemplifica:
    t0 = time.perf_counter()
    file_note = Path(file_note)
    out = Path(cartella_output) if cartella_output else file_note.parent
    dati = json.loads(file_note.read_text(encoding="utf-8"))
    beat = json.loads(Path(file_ritmo).read_text(encoding="utf-8")).get("beat") or []
    nuove, uniti, prima = unisci_note_tenute(dati["note"], beat)
    out.mkdir(parents=True, exist_ok=True)
    file_out = out / "note_semplificate.json"
    file_out.write_text(json.dumps(dict(dati, note=nuove, semplificate=True), ensure_ascii=False), encoding="utf-8")
    avvisi = []
    if prima and uniti / prima > 0.4:
        avvisi.append(f"Semplificazione: uniti {uniti} accordi su {prima} ripetuti mentre suonavano. Se il brano "
                      "ha colpi ripetuti veloci (palm mute), alcuni potrebbero mancare: si può togliere la spunta.")
    return RisultatoSemplifica(str(file_out), prima, uniti, round(time.perf_counter() - t0, 3), avvisi)


@tool
def semplifica_note_audio(file_note: str, file_ritmo: str) -> str:
    """Unisce gli accordi riattaccati mentre stanno ancora suonando (note tenute spezzate dalla trascrizione
    della chitarra distorta). Opzionale: tablatura più pulita, ma qualche colpo ripetuto vero può sparire.
    Restituisce un JSON con 'accordi_uniti' e 'file_note' da passare a crea_tablatura_audio."""
    return json.dumps(asdict(semplifica_note(file_note, file_ritmo)), ensure_ascii=False)
