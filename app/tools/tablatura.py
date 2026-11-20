"""Tool 5 - crea_tablatura: dalle note e dal ritmo alla tablatura (Tablatore).

Passi (tutto codice deterministico, nessun LLM):
  1. Griglia a semicrome costruita sui beat REALI (ritmo.json): segue anche piccole variazioni
     di tempo. Ogni nota va sulla semicroma più vicina.
  2. Battute dai primi tempi e dal metro del tool del ritmo; le note prima del primo tempo
     finiscono in battute aggiunte prima (levare).
  3. Pulizia: doppioni sulla stessa semicroma, massimo 6 note contemporanee.
  4. Power chord (opzionale, spento di default): se in un attacco ci sono solo fondamentale e
     quinta, si aggiunge l'ottava. Misurato su "Nine thou": guadagno minimo sull'audio reale e
     note false sui bicordi a vuoto (fondamentale + quinta suonati davvero senza ottava).
  5. Accordatura: scelta automatica tra le più comuni (o imposta dall'utente).
  6. Corde e tasti con programmazione dinamica: tra tutte le diteggiature possibili si sceglie
     la sequenza con meno salti della mano, meno stiramento e tasti più bassi.
  7. Uscite: tablatura.gp5 (Guitar Pro 5, letto dal visualizzatore alphaTab),
     tablatura.json (per il Verificatore di suonabilità) e tablatura.txt (testo leggibile).

Limiti noti: niente terzine; le note non si legano tra battute (vengono troncate a fine
battuta); la correzione d'ottava non è applicata (misurata non affidabile su audio reale, O33).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config
from app.tools.trascrizione import nome_nota

# durate Guitar Pro espresse in semicrome: (semicrome, valore, puntata)
DURATE = [(16, 1, False), (12, 2, True), (8, 2, False), (6, 4, True), (4, 4, False),
          (3, 8, True), (2, 8, False), (1, 16, False)]


@dataclass
class RisultatoTablatura:
    file_gp5: str
    file_tablatura: str
    file_testo: str
    accordatura: str
    corde: list[str]
    bpm: float
    metro: str
    numero_battute: int
    numero_note: int
    note_scartate: int
    note_aggiunte_power_chord: int
    tasto_max: int
    errore_quantizzazione_ms: float
    tempo_elaborazione_s: float
    avvisi: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- griglia e battute
def _griglia(beat: list[float], t_min: float, t_max: float, metro: int) -> np.ndarray:
    """Beat estesi prima e dopo (con il periodo mediano) fino a coprire tutte le note, con
    almeno una battuta di margine prima (per un'eventuale battuta in levare)."""
    b = list(beat)
    p = float(np.median(np.diff(b)))
    while b[0] > t_min - metro * p:
        b.insert(0, b[0] - p)
    while b[-1] < t_max + metro * p:
        b.append(b[-1] + p)
    return np.array(b)


def _tempi_slot(beat: np.ndarray) -> np.ndarray:
    """Istante di ogni semicroma della griglia."""
    s = config.SUDDIVISIONI_BEAT
    return np.concatenate([beat[i] + (beat[i + 1] - beat[i]) * np.arange(s) / s
                           for i in range(len(beat) - 1)] + [beat[-1:]])


def _calibra_fase(note: list[dict], beat: np.ndarray) -> tuple[np.ndarray, float]:
    """Sposta la griglia dello scarto mediano tra gli attacchi e la semicroma più vicina.
    Misurato su "Nine thou" reale: i beat di beat_this cadono 60-90 ms prima degli attacchi
    (quasi mezza semicroma a 108 BPM), e senza correzione alcune note finivano nella casella
    accanto."""
    slot_t = _tempi_slot(beat)
    passo = float(np.median(np.diff(slot_t)))
    inizi = np.array([n["inizio"] for n in note])
    vicini = slot_t[np.clip(np.searchsorted(slot_t, inizi), 1, len(slot_t) - 1)]
    prima = slot_t[np.clip(np.searchsorted(slot_t, inizi) - 1, 0, len(slot_t) - 1)]
    residui = np.where(np.abs(inizi - prima) < np.abs(inizi - vicini), inizi - prima, inizi - vicini)
    # media circolare: il residuo è definito a meno di un passo della griglia. Il risultato
    # si porta nell'intervallo [-0,3; +0,7) passi, non [-0,5; +0,5): i beat di beat_this
    # cadono PRIMA degli attacchi (misurato: -50 ms sintetizzato, -60/-90 ms reale), e con uno
    # scarto vicino a mezzo passo (+70 ms contro 69) la scelta simmetrica spostava tutto il
    # brano di una semicroma.
    angolo = np.angle(np.mean(np.exp(2j * np.pi * residui / passo)))
    frazione = angolo / (2 * np.pi)
    if frazione < -0.3:
        frazione += 1.0
    spostamento = float(frazione * passo)
    return (beat + spostamento, spostamento) if abs(spostamento) > 0.010 else (beat, 0.0)


def _quantizza(note: list[dict], beat: np.ndarray) -> tuple[list[dict], float]:
    """Assegna a ogni nota la semicroma globale di inizio e la durata in semicrome."""
    slot_t = _tempi_slot(beat)
    errori, out = [], []
    for n in note:
        i = int(np.argmin(np.abs(slot_t - n["inizio"])))
        j = int(np.argmin(np.abs(slot_t - n["fine"])))
        errori.append(abs(slot_t[i] - n["inizio"]))
        out.append(dict(n, slot=i, durata_slot=max(1, j - i)))
    return _risolvi_collisioni(out, slot_t), float(np.median(errori) * 1000) if errori else 0.0


def _risolvi_collisioni(note_q: list[dict], slot_t: np.ndarray) -> list[dict]:
    """Due attacchi DIVERSI nella stessa semicroma non diventano un accordo unico (Step 19).
    Con il tempo dimezzato (Lying From You a 80 BPM) le caselle durano il doppio e accordi vicini
    finivano insieme: eventi di 5-6 note impossibili con la forma drop, 113 note "non suonabili" e
    accordatura sbagliata (C# standard invece di drop C#). Resta l'attacco più vicino alla casella;
    l'altro va nella casella accanto (dalla sua parte) se è libera, altrimenti si scarta.
    Attacchi con le stesse note (riattacco) restano uniti: non creano accordi falsi."""
    per_slot: dict[int, list[list[dict]]] = {}
    for n in sorted(note_q, key=lambda n: n["inizio"]):
        attacchi = per_slot.setdefault(n["slot"], [])
        if attacchi and n["inizio"] - attacchi[-1][0]["inizio"] <= config.FINESTRA_ATTACCO_S:
            attacchi[-1].append(n)
        else:
            attacchi.append([n])
    occupati = set(per_slot)
    out = []
    for s, attacchi in sorted(per_slot.items()):
        if len({frozenset(n["midi"] for n in a) for a in attacchi}) <= 1:
            out += [n for a in attacchi for n in a]
            continue
        attacchi.sort(key=lambda a: abs(a[0]["inizio"] - slot_t[s]))
        out += attacchi[0]
        for a in attacchi[1:]:
            if frozenset(n["midi"] for n in a) == frozenset(n["midi"] for n in attacchi[0]):
                out += a
                continue
            vicino = s + (1 if a[0]["inizio"] > slot_t[s] else -1)
            if 0 <= vicino < len(slot_t) and vicino not in occupati:
                occupati.add(vicino)
                out += [dict(n, slot=vicino, durata_slot=max(1, n["durata_slot"] - (vicino - s))) for n in a]
    return out


# ---------------------------------------------------------------- eventi e power chord
def _eventi(note_q: list[dict]) -> dict[int, list[dict]]:
    """Raggruppa per semicroma; una sola nota per altezza; al massimo 6 (le più intense)."""
    gruppi: dict[int, dict[int, dict]] = {}
    for n in note_q:
        g = gruppi.setdefault(n["slot"], {})
        if n["midi"] not in g or n["intensita"] > g[n["midi"]]["intensita"]:
            g[n["midi"]] = n
    return {s: sorted(g.values(), key=lambda n: -n["intensita"])[:6] for s, g in sorted(gruppi.items())}


def _completa_power_chord(eventi: dict[int, list[dict]]) -> int:
    """Fondamentale + quinta senza ottava -> si aggiunge l'ottava."""
    aggiunte = 0
    for gruppo in eventi.values():
        alt = sorted(n["midi"] for n in gruppo)
        if len(alt) == 2 and alt[1] - alt[0] == 7 and alt[0] + 12 <= config.MIDI_MAX_CHITARRA:
            base = min(gruppo, key=lambda n: n["midi"])
            gruppo.append(dict(base, midi=alt[0] + 12, aggiunta=True))
            aggiunte += 1
    return aggiunte


# ---------------------------------------------------------------- diteggiatura (DP)
def _diteggiature(altezze: list[int], corde: list[int]) -> list[tuple[tuple[int, ...], float, int | None]]:
    """Tutte le assegnazioni nota -> corda distinte e suonabili, con costo statico e posizione.
    corde: MIDI a vuoto dalla 6a (indice 0) alla 1a (indice 5)."""
    risultati = []
    if not altezze:
        return risultati
    for scelta in itertools.permutations(range(6), len(altezze)):
        tasti = [a - corde[c] for a, c in zip(altezze, scelta)]
        if any(t < 0 or t > config.TASTI for t in tasti):
            continue
        premuti = [t for t in tasti if t > 0]
        estensione = (max(premuti) - min(premuti)) if premuti else 0
        if estensione > config.ESTENSIONE_MANO_MAX:
            continue
        usate = sorted(scelta)
        buchi = (usate[-1] - usate[0] + 1 - len(usate)) if len(usate) > 1 else 0
        posizione = min(premuti) if premuti else None
        # Pesi: lo stiramento della mano conta più dell'altezza del tasto. Con 0,1 per tasto
        # (prima versione) i power chord 8-8-8 diventavano "1-3-3" vicino al capotasto:
        # suonabili ma scomodi e diversi a ogni accordo.
        costo = ([0, 0, 0.4, 0.8, 1.5, 3.0][estensione] + 0.03 * (posizione or 0)
                 + 0.15 * max(0, max(tasti) - 12) + 0.5 * buchi)
        # Forma "a un dito" (stesso tasto su 3+ corde vicine, nessuna corda a vuoto): è il power
        # chord canonico delle accordature drop (es. 7-7-7 in drop C#). Senza questo vantaggio il
        # G# usciva 0-2-2 su altre corde (misurato su tutto "Nine Thou"). In accordatura standard
        # i power chord non hanno mai questa forma: nessun effetto.
        if len(tasti) >= 3 and len(premuti) == len(tasti) and len(set(premuti)) == 1 and buchi == 0:
            costo -= config.BONUS_FORMA_A_UN_DITO
        # Ottava "rock" (Step 20, O70): nota grave sulla 6a o 5a corda, ottava due corde sopra, la corda
        # in mezzo stoppata (es. 9-11 sulla 5a e 3a corda nel ritornello di Lying From You). Senza
        # questo vantaggio finiva su 4a + 2a corda vicino al capotasto.
        if (len(altezze) == 2 and abs(altezze[0] - altezze[1]) == 12 and buchi == 1 and len(premuti) == 2
                and scelta[int(np.argmin(altezze))] in (0, 1)):
            costo -= config.BONUS_FORMA_OTTAVA
        risultati.append((scelta, costo, posizione))
    risultati.sort(key=lambda r: r[1])
    return risultati[:40]


def _salto(cand: tuple, cand_prec: tuple) -> float:
    """Costo del passaggio da una diteggiatura alla successiva: spostamento della mano (le
    corde a vuoto non hanno posizione) + cambio di corde tra due accordi (un riff di accordi si
    suona con la stessa forma che scorre sul manico)."""
    scelta, _, pos = cand
    scelta_p, _, pos_p = cand_prec
    costo = 0.2 * abs(pos - pos_p) if pos is not None and pos_p is not None else 0.0
    if len(scelta) >= 2 and len(scelta_p) >= 2 and set(scelta) != set(scelta_p):
        costo += 0.4
    return costo


def _assegna(eventi: list[list[int]], corde: list[int]
             ) -> tuple[list[tuple[list[int], tuple[int, ...]] | None], float, int]:
    """Programmazione dinamica (Viterbi) sulla sequenza degli attacchi.
    Per ogni evento restituisce (altezze suonate, corde scelte) oppure None; poi il costo totale
    e il numero di note impossibili con questa accordatura (scartate, a partire dalla più grave)."""
    passi, fuori = [], 0
    for alt in eventi:
        alt = sorted(alt)
        cand = _diteggiature(alt, corde)
        while not cand and alt:
            alt, fuori = alt[1:], fuori + 1
            cand = _diteggiature(alt, corde)
        passi.append((alt, cand))

    costo_cum: list[list[float]] = []
    indietro: list[list[int]] = []
    for i, (_, cand) in enumerate(passi):
        prec = passi[i - 1][1] if i > 0 else []
        riga_c, riga_i = [], []
        for c in cand:
            costo = c[1]
            if prec:
                valori = [costo_cum[i - 1][k] + _salto(c, cp) for k, cp in enumerate(prec)]
                k = int(np.argmin(valori))
                riga_c.append(costo + valori[k])
                riga_i.append(k)
            else:                                   # inizio di una catena
                riga_c.append(costo)
                riga_i.append(-1)
        costo_cum.append(riga_c)
        indietro.append(riga_i)

    risultato: list = [None] * len(passi)
    totale, k = 0.0, -1
    for i in range(len(passi) - 1, -1, -1):
        alt, cand = passi[i]
        if not cand:
            k = -1
            continue
        if k < 0:                                   # fine di una catena: si sceglie il minimo
            k = int(np.argmin(costo_cum[i]))
            totale += costo_cum[i][k]
        risultato[i] = (alt, cand[k][0])
        k = indietro[i][k]
    return risultato, totale, fuori


# ---------------------------------------------------------------- scrittura
def _scomponi(slot: int) -> list[tuple[int, int, bool]]:
    pezzi = []
    for s, v, p in DURATE:
        while slot >= s:
            pezzi.append((s, v, p))
            slot -= s
    return pezzi


def _scrivi_gp5(percorso: Path, battute: list[dict], corde: list[int], bpm: float, metro: int, titolo: str):
    import guitarpro as gp
    song = gp.Song()
    song.title, song.tempo = titolo, int(round(bpm))
    track = song.tracks[0]
    track.name = "Chitarra"
    track.strings = [gp.GuitarString(i + 1, v) for i, v in enumerate(reversed(corde))]
    track.measures.clear()
    song.measureHeaders.clear()
    start = 960
    for num, b in enumerate(battute, 1):
        h = gp.MeasureHeader(number=num, start=start)
        h.timeSignature = gp.TimeSignature(numerator=metro, denominator=gp.Duration(4))
        song.measureHeaders.append(h)
        start += h.length
        m = gp.Measure(track, h)
        track.measures.append(m)
        voce = m.voices[0]
        for ev in b["eventi"]:
            for k, (_, v, p) in enumerate(_scomponi(ev["durata_slot"])):
                beat = gp.Beat(voce, duration=gp.Duration(v, p),
                               status=gp.BeatStatus.rest if not ev["note"] else gp.BeatStatus.normal)
                for n in ev["note"]:
                    beat.notes.append(gp.Note(beat, value=n["tasto"], string=n["corda"],
                                              type=gp.NoteType.normal if k == 0 else gp.NoteType.tie))
                voce.beats.append(beat)
    gp.write(song, str(percorso))


def _testo(battute: list[dict], corde: list[int], per_riga: int = 4) -> str:
    """Tablatura in testo: una colonna da 3 caratteri per semicroma."""
    nomi = [nome_nota(c)[:-1] for c in corde]           # indice 0 = 6a corda (grave)
    righe_out = []
    for i in range(0, len(battute), per_riga):
        blocco = battute[i:i + per_riga]
        linee = {c: f"{nomi[6 - c]:>2}|" for c in range(1, 7)}   # corda 1 = cantino, in alto
        intest = "   "
        for b in blocco:
            celle = {c: ["---"] * b["slot_battuta"] for c in range(1, 7)}
            for ev in b["eventi"]:
                for n in ev["note"]:
                    celle[n["corda"]][ev["slot"]] = f"{n['tasto']:-<3}"
            for c in range(1, 7):
                linee[c] += "".join(celle[c]) + "|"
            intest += f"{b['numero']:<{3 * b['slot_battuta'] + 1}}"
        righe_out.append(intest.rstrip())
        righe_out += [linee[c] for c in range(1, 7)]
        righe_out.append("")
    return "\n".join(righe_out)


# ---------------------------------------------------------------- funzione principale
def crea_tablatura(file_note: str | Path, file_ritmo: str | Path, accordatura: str = "auto",
                   cartella_output: str | Path | None = None, completa_power_chord: bool = False,
                   titolo: str = "", calibra_fase: bool = True) -> RisultatoTablatura:
    import time
    t0 = time.perf_counter()
    dati_note = json.loads(Path(file_note).read_text(encoding="utf-8"))
    ritmo = json.loads(Path(file_ritmo).read_text(encoding="utf-8"))
    out_dir = Path(cartella_output) if cartella_output else Path(file_note).parent
    note = [n for n in dati_note["note"]]
    if accordatura != "auto" and accordatura not in config.ACCORDATURE:
        raise ValueError(f"accordatura deve essere 'auto' o una tra {list(config.ACCORDATURE)}")
    if not note:
        raise ValueError("Nessuna nota da mettere in tablatura.")
    if not ritmo.get("beat") or len(ritmo["beat"]) < 2:
        raise ValueError("Ritmo non disponibile: servono i beat di analizza_ritmo.")
    avvisi = []
    bpm = float(ritmo["bpm"])
    metro = int(str(ritmo.get("metro") or "4/4").split("/")[0])

    # 1-2. griglia, quantizzazione, battute
    beat = _griglia(ritmo["beat"], min(n["inizio"] for n in note), max(n["fine"] for n in note), metro)
    beat_rete = beat.copy()
    if calibra_fase:
        beat, spostamento = _calibra_fase(note, beat)
    note_q, err_ms = _quantizza(note, beat)
    if err_ms > 40:
        avvisi.append(f"Le note cadono lontano dalla griglia (scarto mediano {err_ms:.0f} ms): ritmo "
                      "approssimativo, possibili terzine o tempo non costante.")
    # indice del beat su cui cade il primo tempo di battuta
    down = ritmo.get("downbeat") or [ritmo["beat"][0]]
    i_down = int(np.argmin(np.abs(beat_rete - down[0])))
    s = config.SUDDIVISIONI_BEAT
    slot_per_battuta = metro * s
    primo_slot = min(n["slot"] for n in note_q)
    inizio_slot = i_down * s
    while inizio_slot > primo_slot:                  # battute in levare prima del primo tempo
        inizio_slot -= slot_per_battuta
    while inizio_slot + slot_per_battuta <= primo_slot:   # salta le battute vuote iniziali
        inizio_slot += slot_per_battuta

    # 3-4. eventi e power chord
    eventi = _eventi(note_q)
    aggiunte = _completa_power_chord(eventi) if completa_power_chord else 0
    slot_ordinati = sorted(eventi)
    altezze = [sorted({n["midi"] for n in eventi[sl]}) for sl in slot_ordinati]

    # 5-6. accordatura e diteggiatura
    # Scelta per priorità (non per somma di costi: con la somma, una nota spuria o piccole
    # differenze di diteggiatura accumulate facevano scegliere "D standard" o "drop D" per un
    # brano in accordatura standard):
    #   1. contano solo le accordature che coprono quasi tutte le note (tolleranza per le note
    #      false: max(2, 2%) note fuori estensione oltre il minimo);
    #   2. tra queste vince la più comune (preferenza);
    #   3. a parità di preferenza decide il costo della diteggiatura.
    candidate = config.ACCORDATURE if accordatura == "auto" else {accordatura: config.ACCORDATURE[accordatura]}
    prove = []
    for nome, (corde, preferenza) in candidate.items():
        scelte, costo, fuori = _assegna([list(a) for a in altezze], corde)
        prove.append((fuori, preferenza, costo, nome, corde, scelte))
    n_note = sum(len(a) for a in altezze)
    tolleranza = max(config.NOTE_FUORI_TOLLERATE, int(config.QUOTA_FUORI_TOLLERATA * n_note))
    minimo = min(p[0] for p in prove)
    ammesse = [p for p in prove if p[0] <= minimo + tolleranza]
    fuori, _, _, nome_acc, corde, scelte = min(ammesse, key=lambda p: (p[1], p[2]))
    if fuori:
        avvisi.append(f"{fuori} note non suonabili con l'accordatura {nome_acc} sono state scartate "
                      "(note false o accordatura diversa: si può indicare l'accordatura a mano).")

    # costruzione delle battute
    eventi_finali = []
    for sl, scelta_ev in zip(slot_ordinati, scelte):
        if scelta_ev is None:
            continue
        alt_ok, scelta = scelta_ev
        dur = max(n["durata_slot"] for n in eventi[sl])
        intens = {n["midi"]: n["intensita"] for n in eventi[sl]}
        note_ev = [{"corda": 6 - c, "tasto": a - corde[c], "midi": a, "nome": nome_nota(a),
                    "intensita": round(float(intens.get(a, 0.0)), 3),
                    "aggiunta": any(n.get("aggiunta") and n["midi"] == a for n in eventi[sl])}
                   for a, c in zip(alt_ok, scelta)]
        eventi_finali.append((sl, dur, sorted(note_ev, key=lambda n: n["corda"])))
    ultimo = max(sl + d for sl, d, _ in eventi_finali)
    n_battute = int(np.ceil((ultimo - inizio_slot) / slot_per_battuta))
    battute = []
    slot_t = _tempi_slot(beat)
    for k in range(n_battute):
        a = inizio_slot + k * slot_per_battuta
        b = a + slot_per_battuta
        dentro = [(sl - a, d, ns) for sl, d, ns in eventi_finali if a <= sl < b]
        ev_b, cursore = [], 0
        for i, (rel, d, ns) in enumerate(dentro):
            if rel > cursore:
                ev_b.append({"slot": cursore, "durata_slot": rel - cursore, "note": []})
            prossimo = dentro[i + 1][0] if i + 1 < len(dentro) else slot_per_battuta
            buco = prossimo - rel
            durata = buco if buco - d < config.PAUSA_MIN_SLOT else min(d, buco)
            ev_b.append({"slot": rel, "durata_slot": durata, "note": ns,
                         "tempo_s": round(float(slot_t[min(a + rel, len(slot_t) - 1)]), 3)})
            cursore = rel + durata
        if cursore < slot_per_battuta:
            ev_b.append({"slot": cursore, "durata_slot": slot_per_battuta - cursore, "note": []})
        battute.append({"numero": k + 1, "inizio_s": round(float(slot_t[min(a, len(slot_t) - 1)]), 3),
                        "slot_battuta": slot_per_battuta, "eventi": ev_b})

    # 7. uscite
    out_dir.mkdir(parents=True, exist_ok=True)
    nomi_corde = [nome_nota(c) for c in corde]
    file_gp5 = out_dir / "tablatura.gp5"
    _scrivi_gp5(file_gp5, battute, corde, bpm, metro, titolo or Path(dati_note.get("sorgente", "")).stem)
    file_json = out_dir / "tablatura.json"
    file_json.write_text(json.dumps({"titolo": titolo or Path(dati_note.get("sorgente", "")).stem,
                                     "accordatura": nome_acc, "corde": nomi_corde, "bpm": bpm,
                                     "metro": f"{metro}/4", "battute": battute}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    file_txt = out_dir / "tablatura.txt"
    file_txt.write_text(f"Accordatura: {nome_acc} ({' '.join(nomi_corde)})  |  ♩ = {bpm:.0f}  |  {metro}/4\n\n"
                        + _testo(battute, corde), encoding="utf-8")

    tutte = [n for b in battute for ev in b["eventi"] for n in ev["note"]]
    return RisultatoTablatura(
        file_gp5=str(file_gp5), file_tablatura=str(file_json), file_testo=str(file_txt),
        accordatura=nome_acc, corde=nomi_corde, bpm=round(bpm, 1), metro=f"{metro}/4",
        numero_battute=len(battute), numero_note=len(tutte), note_scartate=fuori,
        note_aggiunte_power_chord=aggiunte, tasto_max=max((n["tasto"] for n in tutte), default=0),
        errore_quantizzazione_ms=round(err_ms, 1), tempo_elaborazione_s=round(time.perf_counter() - t0, 2),
        avvisi=avvisi,
    )


@tool
def crea_tablatura_audio(file_note: str, file_ritmo: str, accordatura: str = "auto") -> str:
    """Crea la tablatura per chitarra dalle note trascritte ('file_note' di trascrivi_note_audio)
    e dal ritmo ('file_ritmo' di analizza_ritmo_audio). 'accordatura' è 'auto' (scelta automatica)
    oppure una tra: standard, drop D, Eb standard, D standard, drop C#, drop C, C# standard,
    C standard, drop B. Restituisce un JSON riassuntivo con 'accordatura', 'numero_battute',
    'numero_note', 'note_scartate', 'avvisi' e i percorsi 'file_gp5' (Guitar Pro),
    'file_tablatura' (JSON per la verifica di suonabilità) e 'file_testo' (tablatura in testo)."""
    return json.dumps(asdict(crea_tablatura(file_note, file_ritmo, accordatura)), ensure_ascii=False)
