"""Tool 4c - vota_ripetizioni: consenso tra battute che si ripetono (SPERIMENTALE, Step 17).

Nei brani rock i riff si ripetono. Una battuta "ascolta" le sue ripetizioni:
  - impronta di ogni battuta = cromagramma di chitarra e basso separati, a semicrome sulla
    griglia dei beat; due battute sono ripetizioni se le impronte si somigliano (coseno);
  - per ogni nota (altezza + posizione nella battuta) si conta in quante battute del gruppo c'è:
    presente in almeno metà -> aggiunta dove manca; presente in meno di 1/4 -> tolta.
Funziona solo con almeno 2 ripetizioni; le battute senza ripetizioni restano come sono.
Limite misurato: gli errori della trascrizione sono soprattutto SISTEMATICI (uguali in ogni
ripetizione) e il voto non li corregge; su Nine Thou +3-4 punti di richiamo.
Non usa LLM.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config

SLOT_PER_BEAT = 4


@dataclass
class RisultatoRipetizioni:
    file_note: str
    battute: int
    battute_con_ripetizioni: int
    note_aggiunte: int
    note_tolte: int
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)


def griglie(ritmo: dict) -> list[np.ndarray]:
    """Per ogni battuta: i 4 beat + la fine (dai primi tempi e dai beat di analizza_ritmo)."""
    beat, down = np.array(ritmo["beat"]), np.array(ritmo["downbeat"])
    if len(down) < 2:
        return []
    limiti = list(down) + [down[-1] + float(np.median(np.diff(down)))]
    out = []
    for a, b in zip(limiti[:-1], limiti[1:]):
        dentro = beat[(beat >= a - 0.05) & (beat < b - 0.05)]
        g = list(dentro[:4]) if len(dentro) >= 4 else list(np.linspace(a, b, 5)[:4])
        out.append(np.array(g + [b]))
    return out


def _impronte(file_audio, gr, fmin: str) -> np.ndarray:
    """Cromagramma medio per semicroma di ogni battuta, centrato e normalizzato (righe = battute)."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(file_audio)
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    sr, hop = 22050, 256
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz(fmin))
    t = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr, hop_length=hop)
    X = []
    for g in gr:
        pts = np.concatenate([np.linspace(g[k], g[k + 1], SLOT_PER_BEAT + 1)[:-1] for k in range(4)] + [[g[-1]]])
        cols = []
        for a, b in zip(pts[:-1], pts[1:]):
            m = (t >= a) & (t < b)
            cols.append(ch[:, m].mean(axis=1) if m.any() else np.zeros(12))
        v = np.array(cols).T.flatten()
        v = v - v.mean()
        X.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(X)


def _basso_udibile(file_chitarra, file_basso) -> bool:
    """Il basso conta solo se non è molto più debole della chitarra: su una registrazione di sola
    chitarra la sua traccia è silenzio (Lying From You: -99 dB) e il suo cromagramma è rumore, che
    dimezzava la somiglianza tra battute uguali (2 battute ripetute su 108 invece di quasi tutte)."""
    import soundfile as sf
    rms = [float(np.sqrt(np.mean(sf.read(f)[0] ** 2))) + 1e-12 for f in (file_chitarra, file_basso)]
    return 20 * np.log10(rms[1] / rms[0]) > config.BASSO_UDIBILE_DB


def _note_per_battuta(note: list[dict], gr) -> tuple[list[list[dict]], list[dict]]:
    """Note di ogni battuta con la posizione in beat (0-4); le note fuori dalle battute a parte."""
    inizi = np.array([g[0] for g in gr] + [gr[-1][-1]])
    per, fuori = [[] for _ in gr], []
    for n in note:
        a = int(np.searchsorted(inizi, n["inizio"] + 0.035) - 1)
        if not 0 <= a < len(gr):
            fuori.append(n)
            continue
        g = gr[a]
        k = int(np.clip(np.searchsorted(g, n["inizio"] + 0.035) - 1, 0, 3))
        per[a].append(dict(n, _pos=float(k + (n["inizio"] - g[k]) / (g[k + 1] - g[k]))))
    return per, fuori


def _tempo(g, pos: float) -> float:
    k = int(np.clip(np.floor(pos), 0, 3))
    return float(g[k] + (pos - k) * (g[k + 1] - g[k]))


def vota_ripetizioni(file_note: str | Path, file_ritmo: str | Path, file_chitarra: str | Path,
                     file_basso: str | Path, cartella_output: str | Path | None = None) -> RisultatoRipetizioni:
    """Corregge le note con il voto delle battute ripetute; scrive note_ripetizioni.json."""
    t0 = time.perf_counter()
    file_note = Path(file_note)
    out = Path(cartella_output) if cartella_output else file_note.parent
    dati = json.loads(file_note.read_text(encoding="utf-8"))
    note = dati["note"]
    gr = griglie(json.loads(Path(file_ritmo).read_text(encoding="utf-8")))
    if len(gr) < config.GRUPPO_RIPETIZIONI_MIN or not note:
        return RisultatoRipetizioni(str(file_note), len(gr), 0, 0, 0, round(time.perf_counter() - t0, 2),
                                    ["Brano troppo corto per confrontare le ripetizioni."])

    Xc = _impronte(file_chitarra, gr, "C2")
    S = Xc @ Xc.T
    if Path(file_basso).exists() and _basso_udibile(file_chitarra, file_basso):
        Xb = _impronte(file_basso, gr, "C1")
        S = (S + Xb @ Xb.T) / 2
    per, fuori = _note_per_battuta(note, gr)
    tol = config.TOLLERANZA_POSIZIONE_BEAT

    def uguali(n, midi, pos):
        return n["midi"] == midi and abs(n["_pos"] - pos) <= tol

    nuove, aggiunte, tolte, con_gruppo = list(fuori), 0, 0, 0
    for a in range(len(per)):
        vicini = [int(b) for b in np.argsort(-S[a]) if b != a and S[a, b] >= config.SIMILARITA_RIPETIZIONI]
        vicini = vicini[:config.VICINI_RIPETIZIONI_MAX]
        gruppo = [a] + vicini
        if len(gruppo) < config.GRUPPO_RIPETIZIONI_MIN:
            nuove += per[a]
            continue
        con_gruppo += 1

        def supporto(midi, pos):
            return sum(any(uguali(m, midi, pos) for m in per[b]) for b in gruppo) / len(gruppo)

        tenute = []
        for n in per[a]:
            if supporto(n["midi"], n["_pos"]) < config.QUOTA_TOGLI:
                tolte += 1
            else:
                tenute.append(n)
        for b in vicini:
            for m in per[b]:
                if any(uguali(n, m["midi"], m["_pos"]) for n in tenute):
                    continue
                if supporto(m["midi"], m["_pos"]) < config.QUOTA_AGGIUNGI:
                    continue
                simili = [x for c in vicini for x in per[c] if uguali(x, m["midi"], m["_pos"])]
                pos = float(np.median([x["_pos"] for x in simili]))
                inizio = _tempo(gr[a], pos)
                durata = float(np.median([x["fine"] - x["inizio"] for x in simili]))
                tenute.append({"inizio": round(inizio, 4), "fine": round(inizio + durata, 4), "midi": m["midi"],
                               "intensita": float(np.median([x.get("intensita", 0.5) for x in simili])),
                               "_pos": pos, "ripetizioni": True})
                aggiunte += 1
        nuove += tenute

    nuove = sorted(({k: v for k, v in n.items() if k != "_pos"} for n in nuove), key=lambda n: (n["inizio"], n["midi"]))
    file_out = out / "note_ripetizioni.json"
    file_out.write_text(json.dumps(dict(dati, sorgente=f"{dati.get('sorgente', '')} + ripetizioni", note=nuove),
                                   ensure_ascii=False), encoding="utf-8")
    avvisi = []
    if con_gruppo == 0:
        avvisi.append("Nessuna battuta ripetuta riconosciuta: note lasciate come trascritte.")
    return RisultatoRipetizioni(str(file_out), len(gr), con_gruppo, aggiunte, tolte,
                                round(time.perf_counter() - t0, 2), avvisi)


@tool
def vota_ripetizioni_audio(file_note: str, file_ritmo: str, file_chitarra: str, file_basso: str) -> str:
    """Corregge le note confrontando le battute che si ripetono (riff): aggiunge le note presenti
    nella maggior parte delle ripetizioni e toglie quelle quasi assenti. Restituisce un JSON con
    il nuovo 'file_note' da passare al Tablatore, quante note ha aggiunto e tolto, e gli 'avvisi'."""
    return json.dumps(asdict(vota_ripetizioni(file_note, file_ritmo, file_chitarra, file_basso)), ensure_ascii=False)
