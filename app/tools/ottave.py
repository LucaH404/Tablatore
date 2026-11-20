"""Tool 4b-bis - correggi_ottave_sopra: note lette un'ottava sopra su chitarra pulita.

Problema (Step 22, O73, "Dream On"): nell'arpeggio pulito Basic Pitch legge spesso una nota un'ottava
sopra (F4 -> F5): la seconda armonica è più forte della fondamentale, soprattutto nella chitarra
separata per posizione stereo. Il F5 (tasto 13 sulla 1a corda) trascina poi la diteggiatura in alto
e tutto l'arpeggio finisce intorno al 10°-13° tasto invece che al 3°-6°.
Indizi della nota "sopra": nella finestra di ±FINESTRA s la stessa nota un'ottava sotto è suonata
davvero (almeno MIN_SOTTO volte), la nota è più debole della mediana delle vicine ed è tra le più
acute della finestra (sopra l'80° percentile). Allora si abbassa di 12 semitoni.

Si applica SOLO quando i power chord sono spenti (chitarra pulita): nei brani distorti le ottave sono
note vere degli accordi. Misurato:
  - Dream On, parte centro+destra: tasti giusti arpeggio 18% -> 65%, strofe 30% -> 66%, interludio
    70% -> 91%; precisione delle note arpeggio 76% -> 85%. Traccia intera: strofe 80% -> 85%.
  - Applicata anche ai brani distorti peggiorava Lying From You (ritornello accordi 100% -> 90%):
    per questo il vincolo dei power chord spenti.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config
from app.tools.trascrizione import nome_nota


@dataclass
class RisultatoOttave:
    file_note: str
    note_totali: int
    note_corrette: int
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)


def abbassa_ottave_sopra(note: list[dict], finestra: float = config.OTTAVA_SU_FINESTRA_S,
                         min_sotto: int = config.OTTAVA_SU_MIN_SOTTO,
                         percentile: float = config.OTTAVA_SU_PERCENTILE) -> tuple[list[dict], int]:
    """Note con le ottave "sopra" abbassate e numero di note corrette (le decisioni guardano le note originali)."""
    inizi = np.array([n["inizio"] for n in note])
    nuove, corrette = [], 0
    for i, n in enumerate(note):
        idx = [j for j in np.flatnonzero(np.abs(inizi - n["inizio"]) <= finestra) if j != i]
        m = dict(n)
        if idx:
            vicine = [note[j] for j in idx]
            sotto = sum(1 for v in vicine if v["midi"] == n["midi"] - 12)
            debole = n.get("intensita", 1.0) < np.median([v.get("intensita", 1.0) for v in vicine])
            acuta = n["midi"] >= np.percentile([v["midi"] for v in vicine], percentile)
            if sotto >= min_sotto and debole and acuta:
                m["midi"] -= 12
                m["nome"] = nome_nota(m["midi"])
                corrette += 1
        nuove.append(m)
    return nuove, corrette


def correggi_ottave_sopra(file_note: str | Path, cartella_output: str | Path | None = None) -> RisultatoOttave:
    t0 = time.perf_counter()
    file_note = Path(file_note)
    out = Path(cartella_output) if cartella_output else file_note.parent
    dati = json.loads(file_note.read_text(encoding="utf-8"))
    nuove, corrette = abbassa_ottave_sopra(dati["note"])
    out.mkdir(parents=True, exist_ok=True)
    file_out = out / "note_ottave.json"
    file_out.write_text(json.dumps(dict(dati, note=nuove, ottave_corrette=corrette), ensure_ascii=False),
                        encoding="utf-8")
    avvisi = []
    if nuove and corrette / len(nuove) > 0.2:
        avvisi.append(f"Molte note lette un'ottava sopra e abbassate ({corrette} su {len(nuove)}): se la melodia "
                      "suona davvero così in alto, controllare a orecchio le note acute.")
    return RisultatoOttave(str(file_out), len(nuove), corrette, round(time.perf_counter() - t0, 3), avvisi)


@tool
def correggi_ottave_sopra_audio(file_note: str) -> str:
    """Su chitarra pulita (power chord spenti) abbassa di un'ottava le note lette un'ottava sopra: deboli,
    tra le più acute e con la stessa nota un'ottava sotto suonata poco prima o poco dopo.
    Restituisce un JSON con 'note_corrette' e 'file_note' da passare a crea_tablatura_audio."""
    return json.dumps(asdict(correggi_ottave_sopra(file_note)), ensure_ascii=False)
