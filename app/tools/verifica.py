"""Tool 6 - verifica_tablatura: controlla che la tablatura sia suonabile e la corregge (Verificatore).

Controlli (codice deterministico, nessun LLM), evento per evento e tra eventi consecutivi:
  - tasto fuori dal manico (0-24)                                  -> errore
  - due note sulla stessa corda nello stesso istante                -> errore
  - più di 6 note insieme                                           -> errore
  - accordo con la mano troppo aperta (> 5 tasti tra i premuti)     -> errore
  - salto della mano troppo rapido (molti tasti in poco tempo)      -> avviso
  - nota isolata sospetta: molto più acuta delle note vicine e
    senza note vicine nello stesso registro (tipico errore della
    trascrizione: armoniche ed effetti scambiati per note)          -> avviso

Ciclo di correzione (al massimo CICLI_CORREZIONE_MAX giri): verifica -> correzione -> nuova
verifica, finché non restano problemi correggibili.
  - le note isolate sospette e le note impossibili vengono tolte;
  - corde e tasti vengono riassegnati con la stessa programmazione dinamica del Tablatore
    (risolve stessa corda, mano troppo aperta e, dove possibile, i salti).
Le uscite del Tablatore (gp5, json, txt) vengono riscritte come *_verificata.*, l'originale resta.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datapizza.tools import tool

from app import config
from app.tools.tablatura import _assegna, _scrivi_gp5, _testo
from app.tools.trascrizione import nome_nota


@dataclass
class Problema:
    battuta: int
    tempo_s: float
    tipo: str
    gravita: str            # "errore" (non suonabile) o "avviso" (difficile o sospetto)
    descrizione: str


@dataclass
class RisultatoVerifica:
    suonabile: bool
    file_gp5: str
    file_tablatura: str
    file_testo: str
    cicli: int
    problemi_iniziali: int
    problemi_finali: int
    note_tolte: int
    note_riditeggiate: int
    numero_note: int
    tasto_max: int
    tempo_elaborazione_s: float
    problemi: list[dict] = field(default_factory=list)   # quelli rimasti dopo la correzione
    correzioni: list[str] = field(default_factory=list)
    avvisi: list[str] = field(default_factory=list)


def _eventi_con_note(tab: dict) -> list[tuple[int, dict]]:
    return [(b["numero"], ev) for b in tab["battute"] for ev in b["eventi"] if ev["note"]]


def _posizione(note: list[dict]) -> int | None:
    premuti = [n["tasto"] for n in note if n["tasto"] > 0]
    return min(premuti) if premuti else None


# ---------------------------------------------------------------- controlli
def note_sospette(tab: dict) -> set[tuple[int, int]]:
    """Note isolate molto più acute del contesto: (indice evento, corda).
    Una nota è sospetta se supera di almeno SALTO_REGISTRO_SOSPETTO semitoni la mediana delle
    note vicine (entro FINESTRA_CONTESTO_BEAT beat) e nessuna nota vicina sta nel suo registro
    (entro REGISTRO_VICINO semitoni): un assolo acuto ha altre note acute intorno, un'armonica
    spuria no."""
    eventi = _eventi_con_note(tab)
    finestra = config.FINESTRA_CONTESTO_BEAT * 60.0 / tab["bpm"]
    tempi = np.array([ev["tempo_s"] for _, ev in eventi])
    sospette = set()
    for i, (_, ev) in enumerate(eventi):
        vicini = [j for j in np.nonzero(np.abs(tempi - tempi[i]) <= finestra)[0] if j != i]
        altezze_vicine = [n["midi"] for j in vicini for n in eventi[j][1]["note"]]
        if len(altezze_vicine) < 2:
            continue
        mediana = float(np.median(altezze_vicine))
        for n in ev["note"]:
            if (n["midi"] - mediana >= config.SALTO_REGISTRO_SOSPETTO
                    and not any(abs(n["midi"] - a) <= config.REGISTRO_VICINO for a in altezze_vicine)):
                sospette.add((i, n["corda"]))
    return sospette


def controlla(tab: dict) -> list[Problema]:
    problemi = []
    eventi = _eventi_con_note(tab)
    for nb, ev in eventi:
        t, note = ev["tempo_s"], ev["note"]
        if any(not 0 <= n["tasto"] <= config.TASTI for n in note):
            problemi.append(Problema(nb, t, "tasto_fuori_manico", "errore",
                                     f"tasto oltre il {config.TASTI}° o negativo"))
        corde = [n["corda"] for n in note]
        if len(set(corde)) < len(corde):
            problemi.append(Problema(nb, t, "stessa_corda", "errore", "due note sulla stessa corda"))
        if len(note) > 6:
            problemi.append(Problema(nb, t, "troppe_note", "errore", f"{len(note)} note insieme"))
        premuti = [n["tasto"] for n in note if n["tasto"] > 0]
        if premuti and max(premuti) - min(premuti) > config.ESTENSIONE_MANO_MAX:
            problemi.append(Problema(nb, t, "mano_troppo_aperta", "errore",
                                     f"accordo dal tasto {min(premuti)} al {max(premuti)}"))
    # salti della mano: tra due eventi consecutivi con note premute
    for (nb0, e0), (nb1, e1) in zip(eventi, eventi[1:]):
        p0, p1 = _posizione(e0["note"]), _posizione(e1["note"])
        if p0 is None or p1 is None:
            continue
        salto, dt = abs(p1 - p0), e1["tempo_s"] - e0["tempo_s"]
        if salto >= config.SALTO_MANO_MIN_TASTI and salto / max(dt, 1e-3) > config.VELOCITA_MANO_MAX:
            problemi.append(Problema(nb1, e1["tempo_s"], "salto_rapido", "avviso",
                                     f"salto di {salto} tasti in {dt * 1000:.0f} ms"))
    for i, corda in sorted(note_sospette(tab)):
        nb, ev = eventi[i]
        n = next(n for n in ev["note"] if n["corda"] == corda)
        problemi.append(Problema(nb, ev["tempo_s"], "nota_isolata_sospetta", "avviso",
                                 f"{n['nome']} (corda {corda}, tasto {n['tasto']}) molto più acuta "
                                 "delle note vicine: probabile errore di trascrizione"))
    return sorted(problemi, key=lambda p: p.tempo_s)


# ---------------------------------------------------------------- correzione
def _riditeggia(tab: dict, corde: list[int]) -> int:
    """Riassegna corde e tasti a tutti gli eventi (DP del Tablatore). Restituisce quante note
    hanno cambiato posizione; le note impossibili vengono tolte."""
    eventi = _eventi_con_note(tab)
    scelte, _, _ = _assegna([[n["midi"] for n in ev["note"]] for _, ev in eventi], corde)
    cambiate = 0
    for (_, ev), sc in zip(eventi, scelte):
        prima = {(n["midi"], n["corda"]) for n in ev["note"]}
        vecchie = {n["midi"]: n for n in ev["note"]}
        if sc is None:
            ev["note"] = []
            continue
        alt, scelta = sc
        ev["note"] = sorted([dict(vecchie.get(a, {}), corda=6 - c, tasto=a - corde[c], midi=a, nome=nome_nota(a))
                             for a, c in zip(alt, scelta)], key=lambda n: n["corda"])
        cambiate += len({(n["midi"], n["corda"]) for n in ev["note"]} - prima)
    return cambiate


def _togli_sospette(tab: dict) -> list[str]:
    eventi = _eventi_con_note(tab)
    tolte = []
    for i, corda in sorted(note_sospette(tab), reverse=True):
        nb, ev = eventi[i]
        n = next(n for n in ev["note"] if n["corda"] == corda)
        ev["note"] = [m for m in ev["note"] if m is not n]
        tolte.append(f"battuta {nb}: tolta {n['nome']} (tasto {n['tasto']}), nota isolata sospetta")
    return tolte[::-1]


def _unisci_pause(tab: dict):
    """Eventi rimasti senza note: se il buco è corto (< 1 beat) la nota precedente si allunga,
    come fa il Tablatore; altrimenti diventano pause, e pause consecutive si fondono."""
    for b in tab["battute"]:
        out = []
        for ev in b["eventi"]:
            if not ev["note"]:
                ev = {"slot": ev["slot"], "durata_slot": ev["durata_slot"], "note": []}
                if out and (not out[-1]["note"] or ev["durata_slot"] < config.PAUSA_MIN_SLOT):
                    out[-1]["durata_slot"] += ev["durata_slot"]
                    continue
            out.append(ev)
        b["eventi"] = out


# ---------------------------------------------------------------- funzione principale
def verifica_tablatura(file_tablatura: str | Path, correggi: bool = True,
                       cartella_output: str | Path | None = None) -> RisultatoVerifica:
    t0 = time.perf_counter()
    file_tablatura = Path(file_tablatura)
    tab = json.loads(file_tablatura.read_text(encoding="utf-8"))
    if tab.get("accordatura") not in config.ACCORDATURE:
        raise ValueError("Tablatura senza accordatura valida: serve il file JSON di crea_tablatura.")
    if not _eventi_con_note(tab):
        raise ValueError("La tablatura non contiene note.")
    corde = config.ACCORDATURE[tab["accordatura"]][0]
    out_dir = Path(cartella_output) if cartella_output else file_tablatura.parent
    n_iniziali = sum(len(ev["note"]) for _, ev in _eventi_con_note(tab))

    problemi = controlla(tab)
    iniziali, correzioni, riditeggiate, cicli = len(problemi), [], 0, 0
    while correggi and cicli < config.CICLI_CORREZIONE_MAX:
        # i salti rapidi non si correggono: la diteggiatura del Tablatore ha già il minimo di
        # spostamenti, un salto che resta dipende dalle note (vere o false: decide l'orecchio)
        correggibili = [p for p in problemi if p.gravita == "errore" or p.tipo == "nota_isolata_sospetta"]
        if not correggibili:
            break
        cicli += 1
        correzioni += _togli_sospette(tab)
        riditeggiate += _riditeggia(tab, corde)
        _unisci_pause(tab)
        nuovi = controlla(tab)
        if [asdict(p) for p in nuovi] == [asdict(p) for p in problemi]:
            break                                       # nessun progresso: inutile insistere
        problemi = nuovi

    note = [n for _, ev in _eventi_con_note(tab) for n in ev["note"]]
    tolte = n_iniziali - len(note)
    if riditeggiate:
        correzioni.append(f"{riditeggiate} note spostate su un'altra corda/tasto per renderle più comode")
    errori = [p for p in problemi if p.gravita == "errore"]
    avvisi = []
    if errori:
        avvisi.append(f"{len(errori)} punti restano non suonabili: vanno corretti a mano.")
    rimasti_salti = sum(p.tipo == "salto_rapido" for p in problemi)
    if rimasti_salti:
        avvisi.append(f"{rimasti_salti} salti della mano molto rapidi: possibili note sbagliate, da verificare a orecchio.")
    if n_iniziali and tolte / n_iniziali > config.QUOTA_NOTE_TOLTE_ALTA:
        avvisi.append(f"Tolte {tolte} note su {n_iniziali} ({tolte / n_iniziali:.0%}): la trascrizione è poco "
                      "affidabile (effetti, mix denso?).")

    # uscite
    out_dir.mkdir(parents=True, exist_ok=True)
    base = file_tablatura.stem
    file_json = out_dir / f"{base}_verificata.json"
    file_gp5 = out_dir / f"{base}_verificata.gp5"
    file_txt = out_dir / f"{base}_verificata.txt"
    tab["verifica"] = {"note_tolte": tolte, "correzioni": correzioni}
    file_json.write_text(json.dumps(tab, ensure_ascii=False, indent=1), encoding="utf-8")
    metro = int(str(tab.get("metro", "4/4")).split("/")[0])
    _scrivi_gp5(file_gp5, tab["battute"], corde, tab["bpm"], metro, tab.get("titolo", ""))
    file_txt.write_text(f"Accordatura: {tab['accordatura']} ({' '.join(tab['corde'])})  |  ♩ = {tab['bpm']:.0f}  |  "
                        f"{metro}/4  |  verificata\n\n" + _testo(tab["battute"], corde), encoding="utf-8")

    return RisultatoVerifica(
        suonabile=not errori, file_gp5=str(file_gp5), file_tablatura=str(file_json), file_testo=str(file_txt),
        cicli=cicli, problemi_iniziali=iniziali, problemi_finali=len(problemi), note_tolte=tolte,
        note_riditeggiate=riditeggiate, numero_note=len(note), tasto_max=max((n["tasto"] for n in note), default=0),
        tempo_elaborazione_s=round(time.perf_counter() - t0, 3),
        problemi=[asdict(p) for p in problemi], correzioni=correzioni, avvisi=avvisi,
    )


@tool
def verifica_tablatura_audio(file_tablatura: str) -> str:
    """Verifica che la tablatura sia suonabile su una chitarra vera e la corregge: toglie le note
    isolate sospette (probabili errori di trascrizione), sistema corde e tasti impossibili o scomodi.
    'file_tablatura' è il JSON prodotto da crea_tablatura_audio. Restituisce un JSON con 'suonabile',
    'note_tolte', 'problemi_finali', 'avvisi', le prime correzioni e i percorsi dei file verificati
    ('file_gp5', 'file_tablatura', 'file_testo')."""
    r = asdict(verifica_tablatura(file_tablatura))
    r["problemi"] = r["problemi"][:5]                 # riassunto compatto per l'LLM
    r["correzioni"] = r["correzioni"][:5]
    return json.dumps(r, ensure_ascii=False)
