"""Tool 4b - completa_power_chord: ricostruisce i power chord usando la linea di basso.

Problema (misurato su tutto "Nine Thou", 1444 altezze di riferimento dalla tablatura Songsterr):
sulla chitarra distorta Basic Pitch non trova MAI le fondamentali gravi (C#2: 0 volte su 185,
E2: 0 su 38) e dei power chord vede quasi sempre solo ottava o quinta, spesso all'ottava
sbagliata. Senza la fondamentale l'accordatura drop C# non è riconoscibile e le note finiscono
su corde e tasti sbagliati (stessa corda/tasto di Songsterr: 3%).

Idea: nel rock il basso suona la fondamentale insieme ai power chord (misurato: dove il basso
suona, l'84% delle volte la sua nota è la fondamentale, un'ottava sotto). Il basso è monofonico:
la sua nota si stima bene con un metodo classico (CQT + somma delle armoniche; stessa accuratezza
di pYIN, 2,3 s invece di 38 s su 4 minuti). Per ogni attacco della chitarra coerente con il basso
(note = fondamentale o quinta del basso; la terza maggiore è ammessa perché è un'armonica) si
scrive il power chord completo: fondamentale, quinta, ottava.

Attenzione: su un brano NON fatto di power chord la regola è disastrosa (arpeggi puliti del brano
di prova: F1 da 0,88 a 0,17). Per questo in automatico si attiva solo se due indizi indipendenti
sono d'accordo: chitarra distorta (piattezza spettrale) E note della chitarra coerenti con il
basso. Si può sempre forzare ("power_chord") o spegnere ("libero").

Ripiego senza basso (Step 18): su una registrazione di sola chitarra (es. "Lying From You - guitar
only") la traccia del basso è solo il residuo della chitarra e la regola si spegneva, lasciando i
power chord spezzati su corde diverse. Le fondamentali però sono quasi sempre deducibili dalla
chitarra stessa: in ogni accordo trascritto le note stanno in {fondamentale, quinta, (terza come
armonica)} (Lying From You 92% degli accordi, Nine Thou 99%). Se il basso manca o non è coerente,
la chitarra è distorta e gli accordi sono coerenti tra loro, la fondamentale si ricava così. Con il
basso disponibile resta il metodo col basso (misurato su Nine Thou: tab 63,6% contro 61,3%).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config

MODALITA = ("auto", "power_chord", "libero")


@dataclass
class RisultatoPowerChord:
    file_note: str                 # note da usare per la tablatura (originali se la regola è spenta)
    attiva: bool
    modalita: str
    motivo: str
    piattezza: float
    coerenza: float
    quota_basso: float
    attacchi_convertiti: int
    attacchi_totali: int
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)
    fonte: str = ""                # da dove vengono le fondamentali: "basso" o "chitarra"
    coerenza_interna: float = 0.0  # accordi della chitarra che stanno in un power chord


def linea_di_basso(file_basso: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Nota del basso nel tempo (MIDI, NaN dove il basso tace) e quota di tempo in cui suona.
    Il risultato è salvato accanto al file (linea_basso.npz) per non ricalcolarlo."""
    import librosa
    import soundfile as sf
    cache = Path(file_basso).with_name(f"linea_{Path(file_basso).stem}.npz")     # basso.wav -> linea_basso.npz
    if cache.exists() and cache.stat().st_mtime >= Path(file_basso).stat().st_mtime:
        z = np.load(cache)
        return z["t"], z["midi"], float(z["quota"])
    y, sr = sf.read(file_basso)
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=config.SR_BASSO)
    hop, midi0 = config.HOP_BASSO, config.MIDI_MIN_BASSO
    c = np.abs(librosa.cqt(y, sr=config.SR_BASSO, hop_length=hop, fmin=librosa.midi_to_hz(midi0), n_bins=60))
    salienza = c[:36].copy()
    for semitoni, peso in ((12, 0.5), (19, 0.33), (24, 0.25)):     # 2a, 3a e 4a armonica
        salienza += peso * c[semitoni:semitoni + 36]
    energia = 20 * np.log10(salienza.max(axis=0) + 1e-9)
    suona = (energia > np.percentile(energia, 95) - config.DINAMICA_BASSO_DB) & (energia > config.SILENZIO_BASSO_DB)
    midi = np.where(suona, midi0 + np.argmax(salienza, axis=0), np.nan).astype(float)
    t = librosa.frames_to_time(np.arange(len(midi)), sr=config.SR_BASSO, hop_length=hop)
    quota = float(suona.mean()) if len(suona) else 0.0
    np.savez(cache, t=t, midi=midi, quota=quota)
    return t, midi, quota


def piattezza_spettrale(file_chitarra: str | Path) -> float:
    """Piattezza spettrale mediana della chitarra (frame entro 30 dB dal massimo): la distorsione
    la alza molto (misurato: arpeggi puliti 0,0001; chitarre distorte 0,0005-0,0018)."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(file_chitarra)
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    s = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    e = (s ** 2).sum(axis=0)
    if not e.any():
        return 0.0
    attivi = e > np.percentile(e, 95) * 1e-3
    return float(np.median(librosa.feature.spectral_flatness(S=s)[0][attivi]))


def _gruppi(note: list[dict]) -> list[list[dict]]:
    """Note attaccate insieme (entro FINESTRA_ATTACCO_S)."""
    out, cur = [], []
    for n in sorted(note, key=lambda n: n["inizio"]):
        if cur and n["inizio"] - cur[0]["inizio"] > config.FINESTRA_ATTACCO_S:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def _basso_a(t_b, midi_b, t: float) -> int | None:
    sel = (t_b >= t + 0.02) & (t_b <= t + 0.12)          # subito dopo l'attacco
    v = midi_b[sel]
    v = v[~np.isnan(v)]
    return int(round(np.median(v))) if len(v) >= 3 else None


def _coerente(gruppo: list[dict], b: int) -> bool:
    pcs = {n["midi"] % 12 for n in gruppo}
    radice, quinta, terza = b % 12, (b + 7) % 12, (b + 4) % 12
    return pcs <= {radice, quinta, terza} and bool(pcs & {radice, quinta})


def radice_dalla_chitarra(gruppo: list[dict], precedente: int | None = None) -> int | None:
    """Fondamentale (classe di altezza 0-11) del power chord compatibile con le note del gruppo:
    tutte in {fondamentale, quinta, terza maggiore (armonica)} e almeno fondamentale o quinta.
    Una quinta isolata subito dopo un accordo è lo stesso accordo (la fondamentale si perde spesso).
    Tra più candidati vince la fondamentale presente, poi quella della nota più grave."""
    pcs = {n["midi"] % 12 for n in gruppo}
    grave = min(n["midi"] for n in gruppo) % 12
    candidati = [r for r in range(12) if pcs <= {r, (r + 7) % 12, (r + 4) % 12} and pcs & {r, (r + 7) % 12}]
    if not candidati:
        return None
    if len(pcs) == 1 and precedente is not None and (precedente + 7) % 12 in pcs:
        return precedente
    return min(candidati, key=lambda r: (r not in pcs, r != grave))


def coerenza_interna(gruppi: list[list[dict]]) -> tuple[float, int]:
    """Quota degli accordi (2+ altezze diverse) che stanno in un power chord, e quanti accordi ci sono."""
    accordi = [g for g in gruppi if len({n["midi"] % 12 for n in g}) > 1]
    if not accordi:
        return 0.0, 0
    return sum(radice_dalla_chitarra(g) is not None for g in accordi) / len(accordi), len(accordi)


def e_ottava(gruppo: list[dict]) -> bool:
    """Forma a ottava (es. 9-11 sulla 5a e 3a corda nel ritornello di Lying From You): una sola nota
    ripetuta a distanza di ottava, nota più grave in registro medio. Non è un power chord."""
    soglia = config.OTTAVA_MIDI_MIN
    altezze = sorted({n["midi"] for n in gruppo})
    return (soglia is not None and len(altezze) >= 2 and len({a % 12 for a in altezze}) == 1
            and altezze[0] >= soglia)


def _ottave_nel_contesto(gruppi: list[list[dict]], radice_dal_basso=lambda g: False) -> dict[int, list[dict]]:
    """Gruppi da scrivere come ottava (indice -> note). Le ottave arrivano a raffiche di semicrome
    sulla stessa nota: della raffica Basic Pitch sente spesso solo la nota grave o solo l'acuta
    (Lying From You: 61 ottave intere, 44 solo grave, 10 solo acuta). Una nota singola vicina
    (entro FINESTRA_OTTAVA_S) a un'ottava riconosciuta della stessa nota diventa quell'ottava.
    Se il basso (affidabile) suona la stessa nota sotto, è la fondamentale di un power chord di cui
    si sente solo l'ottava (Nine Thou: C#3+C#4 sopra il C# del basso): decide il basso."""
    out = {i: g for i, g in enumerate(gruppi) if e_ottava(g) and not radice_dal_basso(g)}
    piene = sorted(out)
    if not piene:
        return out
    tempi = np.array([gruppi[i][0]["inizio"] for i in piene])
    for i, g in enumerate(gruppi):
        altezze = {n["midi"] for n in g}
        if i in out or len(altezze) != 1 or radice_dal_basso(g):
            continue
        t = g[0]["inizio"]
        k = int(np.argmin(np.abs(tempi - t)))
        if abs(tempi[k] - t) > config.FINESTRA_OTTAVA_S:
            continue
        modello = sorted({n["midi"] for n in gruppi[piene[k]]})
        if next(iter(altezze)) in modello:
            base = g[0]
            out[i] = [dict(base, midi=m, ottava=True) for m in modello]
    return out


def _accordo(g: list[dict], radice: int) -> list[dict]:
    minimo = config.RADICE_POWER_CHORD_MIN
    r = minimo + (radice - minimo) % 12                  # fondamentale nel registro grave della chitarra
    base = min(g, key=lambda n: n["inizio"])
    fine = max(n["fine"] for n in g)
    intensita = max(n.get("intensita", 0.5) for n in g)
    return [dict(base, midi=m, fine=fine, intensita=intensita, power_chord=True) for m in (r, r + 7, r + 12)]


def completa_power_chord(file_note: str | Path, file_basso: str | Path | None, file_chitarra: str | Path,
                         modalita: str = "auto", cartella_output: str | Path | None = None) -> RisultatoPowerChord:
    t0 = time.perf_counter()
    if modalita not in MODALITA:
        raise ValueError(f"modalita deve essere una tra {MODALITA}")
    file_note = Path(file_note)
    out_dir = Path(cartella_output) if cartella_output else file_note.parent
    dati = json.loads(file_note.read_text(encoding="utf-8"))
    note = dati["note"]
    if file_basso and Path(file_basso).exists():
        t_b, midi_b, quota = linea_di_basso(file_basso)
    else:
        t_b, midi_b, quota = np.zeros(0), np.zeros(0), 0.0
    gruppi = _gruppi(note)
    con_basso = [(g, b) for g in gruppi if (b := _basso_a(t_b, midi_b, g[0]["inizio"])) is not None]
    coerenza = sum(_coerente(g, b) for g, b in con_basso) / len(con_basso) if con_basso else 0.0
    interna, n_accordi = coerenza_interna(gruppi)
    piattezza = piattezza_spettrale(file_chitarra)
    basso_ok = quota >= config.QUOTA_BASSO_MIN and len(con_basso) >= config.ATTACCHI_CON_BASSO_MIN
    chitarra_ok = n_accordi >= config.ACCORDI_INTERNI_MIN and interna >= config.COERENZA_INTERNA_MIN
    senza_basso = (f"basso quasi assente ({quota:.0%} del tempo)" if not basso_ok
                   else f"note della chitarra poco coerenti con il basso ({coerenza:.0%})")
    avvisi, fonte = [], ""

    if modalita == "libero":
        attiva, motivo = False, "stile 'note libere' scelto dall'utente"
    elif modalita == "power_chord":
        attiva, motivo = True, "stile 'power chord' scelto dall'utente"
        fonte = "basso" if basso_ok else "chitarra"
    elif piattezza < config.PIATTEZZA_DISTORSIONE_MIN:
        attiva, motivo = False, f"chitarra non distorta (piattezza {piattezza:.4f})"
    elif basso_ok and coerenza >= config.COERENZA_BASSO_MIN:
        attiva, fonte = True, "basso"
        motivo = f"chitarra distorta (piattezza {piattezza:.4f}) e coerente con il basso ({coerenza:.0%})"
    elif chitarra_ok:
        attiva, fonte = True, "chitarra"
        motivo = (f"{senza_basso}; chitarra distorta con accordi coerenti tra loro ({interna:.0%}): "
                  f"fondamentali dedotte dalla chitarra")
    else:
        attiva, motivo = False, senza_basso

    convertiti = 0
    if attiva:
        def radice_dal_basso(g):
            if fonte != "basso":
                return False
            b = _basso_a(t_b, midi_b, g[0]["inizio"])
            return b is not None and b % 12 == g[0]["midi"] % 12
        ottave = _ottave_nel_contesto(gruppi, radice_dal_basso)
        nuove, prec, t_prec = [], None, -1e9
        for i, g in enumerate(gruppi):
            t = g[0]["inizio"]
            if i in ottave:
                nuove += ottave[i]
                prec = None
                continue
            if fonte == "basso":
                b = _basso_a(t_b, midi_b, t)
                r = b % 12 if b is not None and _coerente(g, b) else None
            else:
                r = radice_dalla_chitarra(g, prec if t - t_prec < config.FINESTRA_QUINTA_S else None)
            if r is None:
                nuove += g
                prec = None
                continue
            nuove += _accordo(g, r)
            convertiti += 1
            prec, t_prec = r, t
        out_dir.mkdir(parents=True, exist_ok=True)
        file_out = out_dir / "note_power_chord.json"
        file_out.write_text(json.dumps(dict(dati, note=nuove, power_chord=True), ensure_ascii=False),
                            encoding="utf-8")
        if modalita == "power_chord" and piattezza < config.PIATTEZZA_DISTORSIONE_MIN:
            avvisi.append("Stile 'power chord' forzato su una chitarra che sembra pulita: se il brano è fatto "
                          "di arpeggi o accordi pieni la tablatura peggiora (scegliere 'note libere').")
    else:
        file_out = file_note
    return RisultatoPowerChord(
        file_note=str(file_out), attiva=attiva, modalita=modalita, motivo=motivo, piattezza=round(piattezza, 5),
        coerenza=round(coerenza, 3), quota_basso=round(quota, 3), attacchi_convertiti=convertiti,
        attacchi_totali=len(gruppi), tempo_elaborazione_s=round(time.perf_counter() - t0, 2), avvisi=avvisi,
        fonte=fonte, coerenza_interna=round(interna, 3))


@tool
def completa_power_chord_audio(file_note: str, file_basso: str, file_chitarra: str, modalita: str = "auto") -> str:
    """Ricostruisce i power chord (fondamentale, quinta, ottava) usando la linea di basso o, se il basso manca
    o non è coerente, le note della chitarra stessa: sulla chitarra distorta la trascrizione perde le
    fondamentali gravi. 'modalita': 'auto' (si attiva solo se la chitarra è distorta e coerente con il basso
    o con sé stessa), 'power_chord' (forza) o 'libero' (spenta). Restituisce un JSON con 'attiva', 'motivo',
    'fonte', 'attacchi_convertiti' e 'file_note' da passare a crea_tablatura_audio."""
    return json.dumps(asdict(completa_power_chord(file_note, file_basso, file_chitarra, modalita)), ensure_ascii=False)
