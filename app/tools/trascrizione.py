"""Tool 3 - trascrivi_note: dall'audio della chitarra alle note (Analista audio).

Usa Basic Pitch (Spotify) nel formato ONNX, che gira su CPU senza TensorFlow.
Il modello analizza l'audio una volta; la conversione in note usa soglie tarate
(vedi config.SENSIBILITA) e l'estensione della chitarra come filtro.

Salva due file nella cartella di output:
  - note.json : elenco completo delle note (inizio, fine, MIDI, nome, intensità)
  - note.mid  : file MIDI ascoltabile e importabile in altri programmi
All'agente restituisce solo un RIASSUNTO (numero di note, estensione, densità, polifonia,
avvisi) e il percorso di note.json: l'elenco completo può contenere centinaia di note
e farebbe crescere token e costi senza aiutare l'LLM a decidere.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config

_modello = None
NOMI = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def nome_nota(midi: int) -> str:
    """60 -> 'C4' (Do centrale)."""
    return f"{NOMI[midi % 12]}{midi // 12 - 1}"


def _get_modello():
    global _modello
    if _modello is None:
        # Basic Pitch avvisa per ogni formato di modello non installato (CoreML, TFLite, TF):
        # usiamo solo ONNX, quindi quegli avvisi vengono silenziati durante l'import.
        logging.disable(logging.WARNING)
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model
            _modello = Model(ICASSP_2022_MODEL_PATH)
        finally:
            logging.disable(logging.NOTSET)
    return _modello


@dataclass
class RisultatoTrascrizione:
    file_note: str
    file_midi: str
    numero_note: int
    sensibilita: str
    nota_piu_grave: str | None
    nota_piu_acuta: str | None
    note_al_secondo: float
    note_per_beat: float | None
    polifonia_max: int
    durata_s: float
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)


def _polifonia_max(note: list[dict]) -> int:
    eventi = sorted([(n["inizio"], 1) for n in note] + [(n["fine"], -1) for n in note])
    attive = massimo = 0
    for _, delta in eventi:
        attive += delta
        massimo = max(massimo, attive)
    return massimo


def trascrivi_note(percorso: str | Path, sensibilita: str = "media",
                   cartella_output: str | Path | None = None,
                   bpm: float | None = None) -> RisultatoTrascrizione:
    """Trascrive le note di una traccia di chitarra e le salva in JSON e MIDI.
    Con 'bpm' (dal tool del ritmo) calcola le note per beat e suggerisce la sensibilità."""
    modello = _get_modello()  # prima di tutto: importa basic_pitch con gli avvisi silenziati
    from basic_pitch.inference import run_inference
    from basic_pitch.note_creation import model_output_to_notes, note_events_to_midi

    if sensibilita not in config.SENSIBILITA:
        raise ValueError(f"sensibilita deve essere una tra {list(config.SENSIBILITA)}")
    p = Path(percorso)
    out_dir = Path(cartella_output) if cartella_output else p.parent
    soglie = config.SENSIBILITA[sensibilita]

    t0 = time.perf_counter()
    uscita = run_inference(str(p), modello)
    _, eventi = model_output_to_notes(
        uscita,
        onset_thresh=soglie["onset"],
        frame_thresh=soglie["frame"],
        min_note_len=int(round(soglie["durata_min_ms"] / 1000 * config.FPS_BASIC_PITCH)),
        min_freq=config.FREQ_MIN_HZ,
        max_freq=config.FREQ_MAX_HZ,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )
    durata = len(uscita["note"]) / config.FPS_BASIC_PITCH

    note = sorted(
        ({"inizio": round(float(s), 3), "fine": round(float(e), 3), "midi": int(m),
          "nome": nome_nota(int(m)), "intensita": round(float(a), 3)}
         for s, e, m, a, _ in eventi),
        key=lambda n: (n["inizio"], n["midi"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    file_note = out_dir / "note.json"
    file_note.write_text(json.dumps({"sorgente": p.name, "sensibilita": sensibilita, "note": note},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    file_midi = out_dir / "note.mid"
    note_events_to_midi(eventi, multiple_pitch_bends=False).write(str(file_midi))
    dt = time.perf_counter() - t0

    avvisi = []
    # Densità sulla parte suonata (dal primo all'ultimo attacco): i silenzi iniziali delle
    # registrazioni manuali non devono abbassarla.
    suonato = (note[-1]["fine"] - note[0]["inizio"]) if len(note) > 1 else durata
    densita = len(note) / suonato if suonato > 0 else 0.0
    note_per_beat = round(densita * 60 / bpm, 2) if bpm and note else None
    if not note:
        avvisi.append("Nessuna nota rilevata: l'audio potrebbe essere silenzioso o senza chitarra. "
                      "Si può riprovare con sensibilita='alta'.")
    elif note_per_beat is not None and note_per_beat < config.NOTE_PER_BEAT_MIN and sensibilita != "alta":
        avvisi.append(f"Poche note per beat ({note_per_beat}): con registrazioni reali ed effetti "
                      "molte note sfuggono, riprovare con sensibilita='alta'.")
    elif densita < 0.5 and sensibilita != "alta":
        avvisi.append(f"Poche note ({densita:.2f} al secondo): se ne mancano, riprovare con sensibilita='alta'.")
    elif densita > 12:
        avvisi.append(f"Molte note ({densita:.1f} al secondo): possibili note false, "
                      "riprovare con sensibilita='bassa'.")
    polifonia = _polifonia_max(note)
    if polifonia > 6:
        avvisi.append(f"Fino a {polifonia} note contemporanee: una chitarra ne suona al massimo 6, "
                      "alcune sono probabilmente false.")

    midi_vals = [n["midi"] for n in note]
    return RisultatoTrascrizione(
        file_note=str(file_note),
        file_midi=str(file_midi),
        numero_note=len(note),
        sensibilita=sensibilita,
        nota_piu_grave=nome_nota(min(midi_vals)) if note else None,
        nota_piu_acuta=nome_nota(max(midi_vals)) if note else None,
        note_al_secondo=round(densita, 2),
        note_per_beat=note_per_beat,
        polifonia_max=polifonia,
        durata_s=round(durata, 2),
        tempo_elaborazione_s=round(dt, 2),
        avvisi=avvisi,
    )


@tool
def trascrivi_note_audio(percorso: str, sensibilita: str = "media", bpm: float = 0.0) -> str:
    """Trascrive le note suonate in una traccia audio di chitarra (usare 'audio_consigliato'
    restituito da separa_chitarra_audio). 'sensibilita' può essere 'bassa' (meno note, se ne
    escono troppe), 'media' (default) o 'alta' (più note, se ne mancano). 'bpm' è il tempo
    restituito da analizza_ritmo_audio (0 se non noto): serve a calcolare le note per beat.
    Restituisce un JSON riassuntivo: 'numero_note', 'nota_piu_grave', 'nota_piu_acuta',
    'note_al_secondo', 'note_per_beat', 'polifonia_max', 'avvisi' e 'file_note' (percorso del
    JSON con tutte le note, da passare al passo successivo). Se gli avvisi suggeriscono
    un'altra sensibilità, riprovare una volta."""
    return json.dumps(asdict(trascrivi_note(percorso, sensibilita, bpm=bpm or None)), ensure_ascii=False)
