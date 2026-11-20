"""Tool 4 - analizza_ritmo: tempo (BPM), beat, battute, metro e tonalità (Analista audio).

- Beat e primi tempi di battuta (downbeat): beat_this (JKU Linz, ISMIR 2024, licenza MIT),
  rete neurale che gira su GPU se disponibile. Il tempo si stima sull'audio ORIGINALE,
  perché batteria e basso rendono il ritmo molto più chiaro della sola chitarra.
- Metro (4/4, 3/4...): numero di beat più frequente tra due downbeat consecutivi.
- Tonalità: cromagramma di librosa confrontato con i profili di Krumhansl-Schmuckler.
  Una tonalità maggiore e la sua relativa minore (es. Do maggiore / La minore) hanno le
  stesse note e la stessa armatura: per la tablatura sono equivalenti, quindi il tool
  restituisce sempre anche la relativa.

Salva ritmo.json (con l'elenco completo dei beat, che servirà al Tablatore per costruire
la griglia delle battute) e all'agente restituisce solo un riassunto compatto.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config
from app.tools.audio_io import carica_audio

_rilevatore = None
NOMI_TONALITA = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Profili di Krumhansl-Kessler (1982): quanto ogni grado della scala "appartiene" alla tonalità
PROFILO_MAGGIORE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PROFILO_MINORE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _get_rilevatore():
    global _rilevatore
    if _rilevatore is None:
        import torch
        from beat_this.inference import Audio2Beats
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        _rilevatore = Audio2Beats(checkpoint_path=config.MODELLO_BEAT, device=dispositivo, dbn=False)
        _rilevatore.dispositivo = dispositivo
    return _rilevatore


@dataclass
class RisultatoRitmo:
    file_ritmo: str
    bpm: float | None
    bpm_controllo: float | None
    metro: str | None
    numero_beat: int
    numero_battute: int
    inizio_prima_battuta_s: float | None
    variazione_tempo: float | None
    tonalita: str | None
    tonalita_relativa: str | None
    affidabilita_tonalita: float | None
    durata_s: float
    tempo_elaborazione_s: float
    dispositivo: str
    avvisi: list[str] = field(default_factory=list)
    senza_batteria: bool = False      # beat incoerenti (tipico senza batteria): tempo meno sicuro


def stima_tonalita(audio: np.ndarray, sr: int) -> tuple[str, str, float]:
    """Restituisce (tonalità, relativa, correlazione) col metodo di Krumhansl-Schmuckler."""
    import librosa
    croma = librosa.feature.chroma_cqt(y=audio, sr=sr).mean(axis=1)
    candidati = []
    for i in range(12):
        candidati.append((np.corrcoef(croma, np.roll(PROFILO_MAGGIORE, i))[0, 1], i, "major"))
        candidati.append((np.corrcoef(croma, np.roll(PROFILO_MINORE, i))[0, 1], i, "minor"))
    corr, tonica, modo = max(candidati)
    if modo == "major":
        relativa = f"{NOMI_TONALITA[(tonica + 9) % 12]} minor"
    else:
        relativa = f"{NOMI_TONALITA[(tonica + 3) % 12]} major"
    return f"{NOMI_TONALITA[tonica]} {modo}", relativa, float(corr)


def _periodo_moda(beat: np.ndarray) -> float:
    """Intervallo tra beat più frequente (istogramma a 20 ms = risoluzione di beat_this)."""
    passi, conteggi = np.unique(np.round(np.diff(beat) / 0.02).astype(int), return_counts=True)
    return float(passi[conteggi.argmax()] * 0.02)


def _pulisci_beat(beat: np.ndarray, periodo: float) -> np.ndarray:
    """Scarta i beat troppo vicini al precedente (< 70% del periodo): beat_this sulla sola
    chitarra inserisce beat in più a metà, scambiando le crome per semiminime."""
    tenuti = [beat[0]]
    for b in beat[1:]:
        if b - tenuti[-1] >= 0.7 * periodo:
            tenuti.append(b)
    return np.array(tenuti)


def _completa_beat(beat: np.ndarray, periodo: float) -> np.ndarray:
    """Reinserisce i beat mancanti (salti di 2 o più periodi), equidistanti: la griglia
    delle battute del Tablatore ha bisogno di tutti i beat."""
    completi = [beat[0]]
    for a, b in zip(beat, beat[1:]):
        k = max(1, int(round((b - a) / periodo)))
        completi.extend(a + (b - a) * np.arange(1, k + 1) / k)
    return np.array(completi)


def _griglia_regolare(beat: np.ndarray, periodo: float) -> np.ndarray:
    """Griglia a tempo costante dalla retta di regressione, ricalcolata dopo aver escluso i
    beat sfasati (> 1/4 di periodo dalla prima retta). Usata solo quando beat_this è
    incoerente: senza batteria l'ipotesi più sensata è un tempo costante."""
    indici = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(np.diff(beat) / periodo)))])
    retta = np.polyfit(indici, beat, 1)
    buoni = np.abs(beat - np.polyval(retta, indici)) < 0.25 * periodo
    if buoni.sum() >= 4:
        retta = np.polyfit(indici[buoni], beat[buoni], 1)
    return np.polyval(retta, np.arange(int(indici[-1]) + 1))


def _bpm_regressione(beat: np.ndarray, periodo: float) -> tuple[float, float]:
    """BPM dalla retta che lega l'indice di ogni beat alla sua posizione nel tempo.
    L'indice cresce beat per beat (un beat mancante fa +2), così l'errore non si accumula.
    Più preciso della mediana degli intervalli, quantizzati a 20 ms (140 BPM -> 142,9).
    Restituisce anche la variazione: scarto medio dei beat dalla retta, in frazione di periodo."""
    indici = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(np.diff(beat) / periodo)))])
    pendenza, intercetta = np.polyfit(indici, beat, 1)
    scarto = float(np.std(beat - (pendenza * indici + intercetta)) / pendenza)
    return 60 / float(pendenza), scarto


def _contrasti_accenti(audio: np.ndarray, sr: int, beat: np.ndarray) -> dict[int, tuple[float, int]]:
    """Per raggruppamenti di 2, 3 e 4 beat misura quanto un beat ogni m è più accentato
    degli altri nelle frequenze gravi (< 200 Hz: cassa e basso suonano sul primo tempo).
    Restituisce {m: (contrasto, fase)}: fase = indice del primo beat accentato."""
    import librosa
    hop = 256
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop))
    gravi = S[librosa.fft_frequencies(sr=sr, n_fft=2048) < 200].sum(axis=0)
    attacchi = np.maximum(0, np.diff(gravi, prepend=gravi[0]))
    frame = librosa.time_to_frames(beat, sr=sr, hop_length=hop)
    acc = np.array([attacchi[max(0, i - 2): i + 4].max() for i in frame if i < len(attacchi)])
    acc = acc / (acc.mean() + 1e-9)
    risultato = {}
    for m in (2, 3, 4):
        if len(acc) < 2 * m:
            continue
        risultato[m] = max((float(acc[f::m].mean() - np.delete(acc, np.s_[f::m]).mean()), f)
                           for f in range(m))
    return risultato


def _scegli_metro(contrasti: dict[int, tuple[float, int]]) -> tuple[int, bool]:
    """3/4 solo se il raggruppamento per 3 domina nettamente; altrimenti 4/4, di gran lunga il
    più comune (2/4 e 4/4 hanno gli stessi accenti: si sceglie 4/4). Restituisce (m, certo)."""
    c = {m: v[0] for m, v in contrasti.items()}
    if not c or max(c.values()) < config.CONTRASTO_ACCENTI_MIN:
        return 4, False
    if c.get(3, 0) > config.RAPPORTO_TERZINARIO * max(c.get(2, 0), c.get(4, 0)):
        return 3, True
    return 4, True


def _fase_battute(beat: np.ndarray, downbeat_rete: np.ndarray, m: int, fase_accenti: int) -> int:
    """Su quale beat cade il primo tempo: si usano i downbeat di beat_this se sono coerenti
    con il metro (maggioranza netta sulla stessa fase), altrimenti la fase degli accenti."""
    if len(downbeat_rete):
        vicini = [int(np.argmin(np.abs(beat - d))) % m for d in downbeat_rete]
        fasi, voti = np.unique(vicini, return_counts=True)
        if voti.max() >= 0.6 * len(vicini):
            return int(fasi[voti.argmax()])
    return fase_accenti


def analizza_ritmo(percorso: str | Path, cartella_output: str | Path | None = None) -> RisultatoRitmo:
    """Stima BPM, beat, battute, metro e tonalità; salva tutto in ritmo.json."""
    p = Path(percorso)
    out_dir = Path(cartella_output) if cartella_output else Path(config.CARTELLA_OUTPUT) / p.stem
    rilevatore = _get_rilevatore()

    t0 = time.perf_counter()
    audio = carica_audio(p, config.SAMPLE_RATE_BEAT, canali=1)[0]
    durata = len(audio) / config.SAMPLE_RATE_BEAT
    beat, downbeat_rete = (np.asarray(x, dtype=float) for x in rilevatore(audio, config.SAMPLE_RATE_BEAT))

    avvisi = []
    bpm = bpm_controllo = variazione = metro = None
    irregolare = False
    downbeat = np.array([])
    if len(beat) >= 4:
        import librosa
        periodo = _periodo_moda(beat)
        # Incoerente = molti intervalli diversi dal periodo (es. alterna 0,3 e 0,6 s). Pochi
        # salti doppi (beat mancanti in un'intro senza batteria) non contano: si reinseriscono.
        fuori_periodo = np.abs(np.diff(beat) / periodo - 1) > 0.15
        irregolare = np.mean(fuori_periodo) > config.QUOTA_INTERVALLI_ANOMALI_MAX
        # Secondo parere indipendente (librosa, ~0,1 s): decide il "piano" del tempo quando
        # beat_this è incoerente (sola chitarra: alterna semiminime e crome).
        bpm_controllo = round(float(np.atleast_1d(librosa.feature.tempo(y=audio, sr=config.SAMPLE_RATE_BEAT))[0]), 1)
        if irregolare:
            candidati = (periodo / 2, periodo, periodo * 2)
            periodo = min(candidati, key=lambda c: abs(np.log(c * bpm_controllo / 60)))
            beat = _griglia_regolare(_completa_beat(_pulisci_beat(beat, periodo), periodo), periodo)
            beat = np.clip(beat[beat >= -0.05], 0, None)
            avvisi.append("Beat irregolari (tipico senza batteria): tempo ricostruito a velocità costante "
                          f"con il secondo parere di librosa ({bpm_controllo} BPM). Verificare il tempo a orecchio.")
        else:
            beat = _completa_beat(beat, periodo)
        bpm, variazione = _bpm_regressione(beat, periodo)
        bpm, variazione = round(bpm, 1), round(variazione, 3)
        if variazione > config.VARIAZIONE_TEMPO_MAX:
            avvisi.append(f"Tempo non costante (variazione {variazione:.0%} tra i beat): possibile "
                          "registrazione dal vivo o rubato, la griglia delle battute sarà approssimativa.")
        # Il disaccordo con librosa conta solo se beat_this era incoerente: quando beat_this è
        # coerente è più affidabile (brano reale: beat_this 107,9 = versione sintetizzata 108,1,
        # librosa 143,6).
        rapporto = bpm / bpm_controllo if bpm_controllo else 1.0
        if irregolare and min(abs(np.log(rapporto * k)) for k in (0.5, 1, 2)) > np.log(1.08):
            avvisi.append(f"I due metodi non concordano sul tempo ({bpm} contro {bpm_controllo} BPM): "
                          "tempo incerto.")
        if not config.BPM_MIN_PLAUSIBILE <= bpm <= config.BPM_MAX_PLAUSIBILE:
            avvisi.append(f"BPM {bpm} fuori dall'intervallo plausibile: il tempo potrebbe essere "
                          "stimato al doppio o alla metà.")

        contrasti = _contrasti_accenti(audio, config.SAMPLE_RATE_BEAT, beat)
        beat_per_battuta, certo = _scegli_metro(contrasti)
        if not certo:
            avvisi.append("Accenti poco marcati: metro incerto, si assume 4/4.")
        fase = _fase_battute(beat, downbeat_rete, beat_per_battuta,
                             contrasti.get(beat_per_battuta, (0, 0))[1])
        downbeat = beat[fase::beat_per_battuta]
        metro = f"{beat_per_battuta}/4"
    else:
        avvisi.append("Ritmo non rilevato (meno di 4 beat): l'audio potrebbe essere troppo corto, "
                      "senza pulsazione chiara o silenzioso.")

    tonalita = relativa = None
    affidabilita = None
    if np.sqrt(np.mean(audio ** 2)) > 1e-4:
        tonalita, relativa, affidabilita = stima_tonalita(audio, config.SAMPLE_RATE_BEAT)
        affidabilita = round(affidabilita, 2)
        if affidabilita < 0.6:
            avvisi.append(f"Tonalità incerta (correlazione {affidabilita:.2f}): brano modale, "
                          "cromatico o con poche note.")
    else:
        avvisi.append("Audio silenzioso: tonalità non stimabile.")

    out_dir.mkdir(parents=True, exist_ok=True)
    file_ritmo = out_dir / "ritmo.json"
    file_ritmo.write_text(json.dumps({
        "sorgente": p.name, "bpm": bpm, "metro": metro, "tonalita": tonalita,
        "tonalita_relativa": relativa,
        "beat": [round(float(b), 3) for b in beat],
        "downbeat": [round(float(d), 3) for d in downbeat],
        "senza_batteria": bool(irregolare),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    dt = time.perf_counter() - t0

    return RisultatoRitmo(
        file_ritmo=str(file_ritmo),
        bpm=bpm,
        bpm_controllo=bpm_controllo,
        metro=metro,
        numero_beat=len(beat),
        numero_battute=len(downbeat),
        inizio_prima_battuta_s=round(float(downbeat[0]), 3) if len(downbeat) else None,
        variazione_tempo=variazione,
        tonalita=tonalita,
        tonalita_relativa=relativa,
        affidabilita_tonalita=affidabilita,
        durata_s=round(durata, 2),
        tempo_elaborazione_s=round(dt, 2),
        dispositivo=rilevatore.dispositivo,
        avvisi=avvisi,
        senza_batteria=bool(irregolare),
    )


def ritmo_con_fattore(file_ritmo: str | Path, fattore: float, file_out: str | Path) -> dict:
    """Griglia con il tempo moltiplicato per 'fattore' (0,5 = metà, 2 = doppio), scritta in file_out.
    Step 19: senza batteria il tempo esce spesso doppio (Lying From You: 159 invece di 80): crome a
    159 e semicrome a 80 sono lo stesso ritmo, ma con la griglia doppia i frammenti di una nota
    tenuta finiscono in caselle diverse (note extra) e le battute sono il doppio."""
    r = json.loads(Path(file_ritmo).read_text(encoding="utf-8"))
    beat, downbeat = list(r.get("beat") or []), list(r.get("downbeat") or [])
    if fattore == 0.5 and len(beat) >= 2 and downbeat:
        primo = beat.index(downbeat[0]) if downbeat[0] in beat else 0
        beat, downbeat = beat[primo % 2::2], downbeat[::2]
    elif fattore == 2 and len(beat) >= 2:
        nuovi = []
        for a, b in zip(beat[:-1], beat[1:]):
            nuovi += [a, round((a + b) / 2, 3)]
        nuovi.append(beat[-1])
        primo = nuovi.index(downbeat[0]) if downbeat and downbeat[0] in nuovi else 0
        m = int(str(r.get("metro") or "4/4").split("/")[0])
        beat, downbeat = nuovi, nuovi[primo::m]
    elif fattore != 1:
        raise ValueError(f"fattore del tempo non valido: {fattore}")
    bpm = round(r["bpm"] * fattore, 1) if r.get("bpm") else r.get("bpm")
    nuovo = dict(r, beat=beat, downbeat=downbeat, bpm=bpm, fattore_tempo=fattore)
    Path(file_out).write_text(json.dumps(nuovo, ensure_ascii=False, indent=1), encoding="utf-8")
    return nuovo


@tool
def analizza_ritmo_audio(percorso: str) -> str:
    """Stima tempo (BPM), metro (es. 4/4), battute e tonalità di un brano. Usare il file
    ORIGINALE caricato dall'utente (non la traccia separata della chitarra): batteria e basso
    rendono il ritmo più chiaro. Restituisce un JSON riassuntivo con 'bpm', 'metro',
    'numero_battute', 'tonalita', 'tonalita_relativa' (stesse note, stessa armatura), 'avvisi'
    e 'file_ritmo' (percorso del JSON con tutti i beat, da passare al passo successivo)."""
    return json.dumps(asdict(analizza_ritmo(percorso)), ensure_ascii=False)
