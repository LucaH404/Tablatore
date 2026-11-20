"""Orchestratore: dall'MP3 alla tablatura verificata, con tracciamento di ogni passo.

Due modalità, stessi tool e stesso risultato:
  - "codice" (default): la sequenza fissa dei 6 tool è chiamata dal codice. Le decisioni con
    una regola chiara (ripetere con sensibilità "alta" se le note per beat sono poche, fermarsi
    se il file non è valido o la chitarra manca) sono prese dal codice. L'LLM serve solo alla
    fine, per scrivere un breve resoconto in italiano (una chiamata, facoltativa).
  - "agente": un agente datapizza-ai con i 6 tool decide da solo la sequenza (modalità usata
    per le prove e per il confronto; costa ~15-20 volte più token).

Guardrail (O27): gli avvisi dei tool sono raccolti DAL CODICE e restituiti sempre all'utente,
anche in modalità agente (si leggono dai risultati dei tool registrati nella traccia), senza
dipendere da cosa l'LLM decide di riportare. Gli errori di un passo fermano la pipeline con un
messaggio chiaro; un errore dell'LLM nel resoconto non blocca la tablatura.

Osservabilità: ContextTracing di datapizza-ai (OpenTelemetry). Ogni passo è uno span con
durata, avvisi, decisioni ed errori; ogni chiamata al modello è uno span con i token. La
traccia è salvata in traccia.json nella cartella del brano.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datapizza.tracing import ContextTracing
from datapizza.tracing.tracing import tool_span

from app import config
from app.tools.power_chord import MODALITA as MODALITA_POWER_CHORD
from app.tools.ottave import correggi_ottave_sopra
from app.tools.power_chord import completa_power_chord
from app.tools.ripetizioni import vota_ripetizioni
from app.tools.ritmo import analizza_ritmo
from app.tools.ritmo import ritmo_con_fattore
from app.tools.semplifica import semplifica_note
from app.tools.separazione import separa_chitarra
from app.tools.tablatura import crea_tablatura
from app.tools.trascrizione import trascrivi_note
from app.tools.validazione import valida_file
from app.tools.verifica import verifica_tablatura

AVVISI_FISSI = [config.AVVISO_APPROSSIMAZIONE, config.AVVISO_COPYRIGHT]


class _Stop(Exception):
    """Fermata prevista della pipeline (file non valido, chitarra assente...): non è un errore."""


@dataclass
class RisultatoBrano:
    completato: bool
    modalita: str
    motivo_stop: str | None = None
    file_gp5: str | None = None
    file_tablatura: str | None = None
    file_testo: str | None = None
    file_midi: str | None = None
    file_traccia: str | None = None
    accordatura: str | None = None
    bpm: float | None = None
    metro: str | None = None
    tonalita: str | None = None
    sensibilita: str | None = None
    numero_battute: int | None = None
    numero_note: int | None = None
    note_tolte_verifica: int | None = None
    suonabile: bool | None = None
    resoconto: str | None = None
    avvisi: list[str] = field(default_factory=list)
    avvisi_per_passo: dict[str, list[str]] = field(default_factory=dict)
    decisioni: list[str] = field(default_factory=list)
    passi: list[dict] = field(default_factory=list)
    cartella: str | None = None
    titolo: str | None = None
    stile: str = "auto"
    file_note: str | None = None             # note usate per la tablatura (con o senza power chord)
    power_chord: dict = field(default_factory=dict)
    separatore: str = "demucs"               # usato davvero (RoFormer può ripiegare su Demucs)
    ripetizioni: dict = field(default_factory=dict)
    bpm_rilevato: float | None = None        # tempo misurato; bpm = bpm_rilevato * fattore_tempo
    fattore_tempo: float = 1.0               # 0,5 = tempo dimezzato (auto senza batteria o scelto)
    semplifica: dict = field(default_factory=dict)
    token_in: int = 0
    token_out: int = 0
    costo_usd: float = 0.0
    tempo_totale_s: float = 0.0


def _unisci_avvisi(*liste) -> list[str]:
    """Avvisi senza doppioni, nell'ordine in cui arrivano; quelli fissi sempre in fondo."""
    visti, out = set(AVVISI_FISSI), []
    for lista in liste:
        for a in lista or []:
            if a not in visti:
                visti.add(a)
                out.append(a)
    return out + AVVISI_FISSI


def _costo(token_in: int, token_out: int, modello: str = config.MODELLO_LLM) -> float:
    p_in, p_out = config.PREZZI_LLM.get(modello, (0.0, 0.0))
    return round((token_in * p_in + token_out * p_out) / 1e6, 6)


def _salva_traccia(spans, file_traccia: Path, inizio_ns: int) -> tuple[list[dict], int, int]:
    """Converte gli span in un elenco leggibile e somma i token delle chiamate al modello."""
    righe, t_in, t_out = [], 0, 0
    for s in sorted(spans, key=lambda s: s.start_time):
        a = dict(s.attributes or {})
        riga = {"nome": s.name, "tipo": a.get("type"), "inizio_s": round((s.start_time - inizio_ns) / 1e9, 3),
                "durata_s": round((s.end_time - s.start_time) / 1e9, 3)}
        for k in ("decisione", "errore", "esito", "model_name", "prompt_tokens_used", "completion_tokens_used"):
            if k in a:
                riga[k] = a[k]
        if "avvisi" in a:
            riga["avvisi"] = list(a["avvisi"])
        if a.get("type") == "generation":
            t_in += int(a.get("prompt_tokens_used", 0))
            t_out += int(a.get("completion_tokens_used", 0))
        righe.append(riga)
    file_traccia.parent.mkdir(parents=True, exist_ok=True)
    file_traccia.write_text(json.dumps({"spans": righe, "token_in": t_in, "token_out": t_out,
                                        "costo_usd": _costo(t_in, t_out)}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return righe, t_in, t_out


class _Passo:
    """Span di un passo: registra durata, avvisi, decisioni ed eventuale errore."""

    def __init__(self, nome: str, su_passo=None):
        self.nome = nome
        self.su_passo = su_passo

    def __enter__(self):
        if self.su_passo:                        # avanzamento per la pagina web
            try:
                self.su_passo(self.nome)
            except Exception:
                pass
        self._cm = tool_span(f"passo {self.nome}")
        self.span = self._cm.__enter__()
        return self

    def registra(self, avvisi=None, decisione: str | None = None, **attributi):
        if avvisi:
            self.span.set_attribute("avvisi", list(avvisi))
        if decisione:
            self.span.set_attribute("decisione", decisione)
        for k, v in attributi.items():
            self.span.set_attribute(k, v)

    def __exit__(self, tipo, valore, tb):
        if valore is not None:
            self.span.set_attribute("errore", f"{tipo.__name__}: {valore}")
            self.span.set_attribute("esito", "errore")
            self.span.record_exception(valore)
        else:
            self.span.set_attribute("esito", "ok")
        return self._cm.__exit__(tipo, valore, tb)


def _client_llm():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from datapizza.clients.openai import OpenAIClient
    return OpenAIClient(api_key=key, model=config.MODELLO_LLM)


PROMPT_RESOCONTO = (
    "Sei un insegnante di chitarra. Ti do in JSON il risultato di una trascrizione automatica "
    "MP3 -> tablatura. Scrivi in italiano, in 3-4 frasi semplici per un chitarrista, testo semplice "
    "senza grassetto né elenchi: cosa è stato trovato (accordatura, tempo, tonalità, battute), quanto "
    "fidarsi del risultato e cosa controllare a orecchio. Usa SOLO i dati forniti: non inventare note, "
    "accordi o numeri. Non ripetere gli avvisi parola per parola: vengono già mostrati a parte.\n\nDati:\n"
)


def _dati_resoconto(r: RisultatoBrano) -> dict:
    # nomi espliciti: con "note_tolte_verifica" l'LLM aveva scritto "7 note da controllare"
    return {"accordatura": r.accordatura, "bpm": r.bpm, "metro": r.metro, "tonalita": r.tonalita,
            "sensibilita_trascrizione": r.sensibilita, "battute": r.numero_battute,
            "note_in_tablatura": r.numero_note,
            "note_false_gia_tolte_dal_verificatore": r.note_tolte_verifica,
            "tablatura_suonabile": r.suonabile, "decisioni_automatiche": r.decisioni,
            "avvisi": [a for a in r.avvisi if a not in AVVISI_FISSI]}


def _numeri(testo: str) -> set[float]:
    import re
    return {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", testo)}


def numeri_inventati(testo: str, dati: dict) -> set[float]:
    """Guardrail sull'uscita dell'LLM: numeri del resoconto che non compaiono nei dati.
    Sono ammessi i numeri piccoli (1-6: "due salti", "4/4", "6 corde") e gli arrotondamenti
    all'intero (107,9 BPM -> "108")."""
    validi = _numeri(json.dumps(dati, ensure_ascii=False))
    validi |= {float(round(v)) for v in validi} | {float(i) for i in range(7)}
    return {n for n in _numeri(testo) if n not in validi}


def resoconto_di_riserva(r: RisultatoBrano) -> str:
    """Resoconto scritto dal codice: usato se l'LLM non è disponibile o inventa numeri."""
    testo = (f"Accordatura {r.accordatura}, circa {round(r.bpm or 0)} BPM in {r.metro}, tonalità {r.tonalita}: "
             f"{r.numero_battute} battute con {r.numero_note} note.")
    if r.note_tolte_verifica:
        testo += f" Il Verificatore ha tolto {r.note_tolte_verifica} note probabilmente false."
    return testo + " Controlla a orecchio soprattutto i punti indicati negli avvisi."


def _resoconto(r: RisultatoBrano) -> tuple[str, bool, str]:
    """(testo da mostrare, scritto dall'LLM?, testo originale dell'LLM)"""
    client = _client_llm()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY mancante")
    dati = _dati_resoconto(r)
    testo = (client.invoke(PROMPT_RESOCONTO + json.dumps(dati, ensure_ascii=False)).text or "").strip()
    if not testo or numeri_inventati(testo, dati):
        return resoconto_di_riserva(r), False, testo
    return testo, True, testo


# ---------------------------------------------------------------- modalità "codice"
def trascrivi_brano(percorso: str | Path, accordatura: str = "auto", sensibilita: str = "auto",
                    cartella_output: str | Path | None = None, resoconto_llm: bool = True,
                    modalita: str = "codice", su_passo=None, titolo: str | None = None,
                    stile: str = "auto", separatore: str = "demucs", ripetizioni: bool = False,
                    semplifica: bool = False, tempo: float | None = None) -> RisultatoBrano:
    """Pipeline completa. 'sensibilita': "auto" (media, poi alta se servono più note) oppure
    una tra bassa/media/alta; 'accordatura': "auto" o una di config.ACCORDATURE.
    'separatore' ("demucs"/"roformer") e 'ripetizioni' (passo 4c) sono opzioni sperimentali.
    'semplifica' (passo 4d) unisce gli accordi riattaccati mentre suonano. 'tempo': fattore rispetto
    al tempo rilevato (0,5 / 1 / 2) oppure None = automatico (dimezza senza batteria sopra 130 BPM).
    'su_passo(nome)' è chiamata all'inizio di ogni passo (avanzamento per la pagina web)."""
    if modalita == "agente":
        return trascrivi_brano_agente(percorso, cartella_output)
    if sensibilita != "auto" and sensibilita not in config.SENSIBILITA:
        raise ValueError(f"sensibilita deve essere 'auto' o una tra {list(config.SENSIBILITA)}")
    if accordatura != "auto" and accordatura not in config.ACCORDATURE:
        raise ValueError(f"accordatura deve essere 'auto' o una tra {list(config.ACCORDATURE)}")
    if stile not in MODALITA_POWER_CHORD:
        raise ValueError(f"stile deve essere uno tra {MODALITA_POWER_CHORD}")
    if separatore not in config.SEPARATORI:
        raise ValueError(f"separatore deve essere uno tra {config.SEPARATORI}")
    if tempo is not None and tempo not in config.FATTORI_TEMPO:
        raise ValueError(f"tempo deve essere None (automatico) o uno tra {config.FATTORI_TEMPO}")
    p = Path(percorso)
    out = Path(cartella_output) if cartella_output else Path(config.CARTELLA_OUTPUT) / p.stem
    r = RisultatoBrano(completato=False, modalita="codice", cartella=str(out), titolo=titolo or p.stem)
    av = r.avvisi_per_passo
    t0 = time.perf_counter()
    inizio_ns = time.time_ns()

    with ContextTracing().trace(f"trascrivi_brano {p.name}") as traccia:
        try:
            with _Passo("1 validazione", su_passo) as passo:
                v = valida_file(p)
                av["validazione"] = v.avvisi
                passo.registra(v.avvisi + v.errori)
            if not v.valido:
                r.motivo_stop = "File non valido: " + "; ".join(v.errori)
                raise _Stop

            with _Passo("2 separazione", su_passo) as passo:
                s = separa_chitarra(p, out, separatore)
                av["separazione"] = s.avvisi
                d = None
                if s.separatore != separatore:
                    d = f"Separatore {separatore} non utilizzabile su questo brano: usato {s.separatore}."
                    r.decisioni.append(d)
                passo.registra(s.avvisi, decisione=d, quota_chitarra=s.quota_chitarra, separatore=s.separatore)
            r.separatore = s.separatore
            if not s.chitarra_presente:
                r.motivo_stop = s.avvisi[0] if s.avvisi else "Chitarra non presente nel brano."
                raise _Stop

            with _Passo("3 ritmo", su_passo) as passo:
                rit = analizza_ritmo(p, out)              # sul file ORIGINALE (Step 8)
                av["ritmo"] = rit.avvisi
                passo.registra(rit.avvisi, bpm=rit.bpm or 0.0)
            r.bpm, r.metro, r.bpm_rilevato = rit.bpm, rit.metro, rit.bpm
            r.tonalita = f"{rit.tonalita} (relativa {rit.tonalita_relativa})" if rit.tonalita else None
            if tempo is None:
                tempo = 1.0
                if rit.senza_batteria and rit.bpm and rit.bpm > config.BPM_DIMEZZA_SENZA_BATTERIA:
                    tempo = 0.5
                    av["tempo"] = [f"Tempo dimezzato in automatico ({rit.bpm} → {round(rit.bpm / 2, 1)} BPM): senza "
                                   "batteria il tempo rilevato è spesso il doppio. Se è sbagliato, scegliere "
                                   "«Tempo» → rilevato e rifare la tablatura."]
            _ritmo_tablatura(r, out, tempo)

            with _Passo("4 trascrizione", su_passo) as passo:
                prima = "media" if sensibilita == "auto" else sensibilita
                tr = trascrivi_note(s.audio_consigliato, prima, out, bpm=rit.bpm)
                if (sensibilita == "auto" and tr.note_per_beat is not None
                        and tr.note_per_beat < config.NOTE_PER_BEAT_MIN):
                    d = (f"Poche note per beat con sensibilità media ({tr.note_per_beat:.2f} < "
                         f"{config.NOTE_PER_BEAT_MIN}): trascrizione ripetuta con sensibilità alta.")
                    r.decisioni.append(d)
                    tr = trascrivi_note(s.audio_consigliato, "alta", out, bpm=rit.bpm)
                    passo.registra(decisione=d)
                av["trascrizione"] = tr.avvisi  # avvisi dell'ultima trascrizione, quella usata
                passo.registra(tr.avvisi, numero_note=tr.numero_note, sensibilita=tr.sensibilita)
            r.sensibilita, r.file_midi = tr.sensibilita, tr.file_midi
            if tr.numero_note == 0:
                r.motivo_stop = "Nessuna nota trascritta: la chitarra non è riconoscibile."
                raise _Stop

            _power_chord(r, out, stile, su_passo)
            if ripetizioni:
                _ripetizioni(r, out, su_passo)
            if semplifica:
                _semplifica(r, out, su_passo)
            _tablatura_e_verifica(r, out, accordatura, su_passo)
        except _Stop:
            pass
        except Exception as e:                       # errore di un passo: già registrato nello span
            r.motivo_stop = f"Errore interno: {type(e).__name__}: {e}"

        r.avvisi = _unisci_avvisi(*av.values())
        if r.completato:
            _scrivi_resoconto(r, resoconto_llm, su_passo)
        r.passi, r.token_in, r.token_out = _salva_traccia(traccia.get_spans(), out / "traccia.json", inizio_ns)

    r.file_traccia = str(out / "traccia.json")
    r.costo_usd = _costo(r.token_in, r.token_out)
    r.tempo_totale_s = round(time.perf_counter() - t0, 2)
    return r


def _power_chord(r: RisultatoBrano, out: Path, stile: str, su_passo=None):
    """Passo 4b: power chord ricostruiti con il basso o, se il basso manca, dalla chitarra stessa."""
    r.stile = stile
    r.file_note = str(out / "note.json")
    basso = out / "basso.wav"
    with _Passo("4b power chord", su_passo) as passo:
        pc = completa_power_chord(out / "note.json", basso if basso.exists() else None, out / "chitarra.wav",
                                  stile, out)
        r.avvisi_per_passo["power chord"] = pc.avvisi
        passo.registra(pc.avvisi, decisione=f"power chord {'attivi' if pc.attiva else 'spenti'}: {pc.motivo}",
                       attacchi_convertiti=pc.attacchi_convertiti, fonte=pc.fonte)
    r.file_note = pc.file_note
    r.power_chord = {"attiva": pc.attiva, "motivo": pc.motivo, "attacchi_convertiti": pc.attacchi_convertiti,
                     "attacchi_totali": pc.attacchi_totali, "fonte": pc.fonte}
    r.decisioni = [d for d in r.decisioni if not d.startswith(("Power chord", "Note un'ottava sopra"))]
    r.avvisi_per_passo.pop("ottave", None)
    if pc.attiva:
        da = "con il basso" if pc.fonte == "basso" else "dalle note della chitarra"
        r.decisioni.append(f"Power chord ricostruiti {da} ({pc.motivo}): {pc.attacchi_convertiti} "
                           f"attacchi su {pc.attacchi_totali}.")
        return
    r.decisioni.append(f"Power chord non ricostruiti: {pc.motivo}.")
    # Chitarra pulita: le note lette un'ottava sopra si abbassano (Step 22). Nei brani distorti no:
    # lì le ottave sono note vere degli accordi.
    with _Passo("4b ottave", su_passo) as passo:
        ott = correggi_ottave_sopra(r.file_note, out)
        r.avvisi_per_passo["ottave"] = ott.avvisi
        passo.registra(ott.avvisi, note_corrette=ott.note_corrette, note_totali=ott.note_totali)
    r.file_note = ott.file_note
    if ott.note_corrette:
        r.decisioni.append(f"Note un'ottava sopra abbassate (chitarra pulita): {ott.note_corrette} su "
                           f"{ott.note_totali}.")


def _file_ritmo(out: Path) -> Path:
    """Griglia per i passi 4c-5: quella con il fattore del tempo, se c'è (lavori vecchi: ritmo.json)."""
    f = out / "ritmo_tablatura.json"
    return f if f.exists() else out / "ritmo.json"


def _ritmo_tablatura(r: RisultatoBrano, out: Path, fattore: float):
    """Scrive ritmo_tablatura.json con il tempo moltiplicato per 'fattore' e aggiorna bpm e decisioni."""
    nuovo = ritmo_con_fattore(out / "ritmo.json", fattore, out / "ritmo_tablatura.json")
    r.bpm_rilevato = r.bpm_rilevato or json.loads((out / "ritmo.json").read_text(encoding="utf-8")).get("bpm")
    r.fattore_tempo, r.bpm = fattore, nuovo["bpm"]
    r.decisioni = [d for d in r.decisioni if not d.startswith("Tempo")]
    if fattore != 1:
        come = "dimezzato" if fattore < 1 else "raddoppiato"
        r.decisioni.append(f"Tempo {come}: {r.bpm_rilevato} → {r.bpm} BPM.")


def _semplifica(r: RisultatoBrano, out: Path, su_passo=None):
    """Passo 4d (opzionale): accordi riattaccati mentre suonano -> uno solo."""
    with _Passo("4d semplifica", su_passo) as passo:
        sp = semplifica_note(r.file_note or out / "note.json", _file_ritmo(out), out)
        r.avvisi_per_passo["semplifica"] = sp.avvisi
        passo.registra(sp.avvisi, accordi_uniti=sp.accordi_uniti, accordi_prima=sp.accordi_prima)
    r.file_note = sp.file_note
    r.semplifica = {"attiva": True, "accordi_uniti": sp.accordi_uniti, "accordi_prima": sp.accordi_prima}
    r.decisioni = [d for d in r.decisioni if not d.startswith("Note ripetute")]
    r.decisioni.append(f"Note ripetute semplificate: {sp.accordi_uniti} accordi su {sp.accordi_prima} erano "
                       "riattacchi dello stesso accordo mentre suonava.")


def _ripetizioni(r: RisultatoBrano, out: Path, su_passo=None):
    """Passo 4c (sperimentale): consenso tra battute ripetute, sulle note del passo 4b."""
    with _Passo("4c ripetizioni", su_passo) as passo:
        rp = vota_ripetizioni(r.file_note or out / "note.json", _file_ritmo(out), out / "chitarra.wav",
                              out / "basso.wav", out)
        r.avvisi_per_passo["ripetizioni"] = rp.avvisi
        passo.registra(rp.avvisi, note_aggiunte=rp.note_aggiunte, note_tolte=rp.note_tolte,
                       battute_con_ripetizioni=rp.battute_con_ripetizioni)
    r.file_note = rp.file_note
    r.ripetizioni = {"attiva": True, "battute": rp.battute, "battute_con_ripetizioni": rp.battute_con_ripetizioni,
                     "note_aggiunte": rp.note_aggiunte, "note_tolte": rp.note_tolte}
    r.decisioni = [d for d in r.decisioni if not d.startswith("Confronto tra ripetizioni")]
    r.decisioni.append(f"Confronto tra ripetizioni: {rp.battute_con_ripetizioni} battute su {rp.battute} hanno "
                       f"ripetizioni; {rp.note_aggiunte} note aggiunte e {rp.note_tolte} tolte.")


def _tablatura_e_verifica(r: RisultatoBrano, out: Path, accordatura: str, su_passo=None):
    """Passi 5-6, usati anche da riaccorda()."""
    with _Passo("5 tablatura", su_passo) as passo:
        file_note = Path(r.file_note) if r.file_note else out / "note.json"
        tab = crea_tablatura(file_note, _file_ritmo(out), accordatura, out, titolo=r.titolo or "")
        r.avvisi_per_passo["tablatura"] = tab.avvisi
        passo.registra(tab.avvisi, accordatura=tab.accordatura, numero_battute=tab.numero_battute)
    r.accordatura = tab.accordatura

    with _Passo("6 verifica", su_passo) as passo:
        ver = verifica_tablatura(tab.file_tablatura, cartella_output=out)
        r.avvisi_per_passo["verifica"] = ver.avvisi
        passo.registra(ver.avvisi, note_tolte=ver.note_tolte, suonabile=ver.suonabile)
    r.file_gp5, r.file_tablatura, r.file_testo = ver.file_gp5, ver.file_tablatura, ver.file_testo
    r.numero_battute, r.numero_note = tab.numero_battute, ver.numero_note
    r.note_tolte_verifica, r.suonabile = ver.note_tolte, ver.suonabile
    r.completato = True


def _scrivi_resoconto(r: RisultatoBrano, resoconto_llm: bool, su_passo=None):
    r.resoconto = resoconto_di_riserva(r)
    if not resoconto_llm:
        return
    try:
        with _Passo("7 resoconto LLM", su_passo) as passo:
            testo, da_llm, _ = _resoconto(r)
            if not da_llm:
                passo.registra(decisione="resoconto LLM scartato (numeri non presenti nei dati): "
                                         "usato il testo di riserva")
            r.resoconto = testo
    except Exception as e:
        r.avvisi.insert(0, f"Resoconto automatico non disponibile ({e}): la tablatura è comunque completa.")


def riaccorda(r: RisultatoBrano, accordatura: str, resoconto_llm: bool = False,
              stile: str | None = None, ripetizioni: bool | None = None, semplifica: bool | None = None,
              tempo: float | None = None) -> RisultatoBrano:
    """Rifà solo tablatura e verifica con un'altra accordatura (e, se indicato, un altro stile, il
    confronto tra ripetizioni o la semplificazione accesi/spenti, un altro fattore del tempo: ripete
    anche i passi 4b-4d), riusando note, ritmo e basso già calcolati (0,1-5 s invece di 10-40 s).
    Utile quando le scelte automatiche sbagliano (O38)."""
    import copy
    if accordatura != "auto" and accordatura not in config.ACCORDATURE:
        raise ValueError(f"accordatura deve essere 'auto' o una tra {list(config.ACCORDATURE)}")
    if stile is not None and stile not in MODALITA_POWER_CHORD:
        raise ValueError(f"stile deve essere uno tra {MODALITA_POWER_CHORD}")
    if tempo is not None and tempo not in config.FATTORI_TEMPO:
        raise ValueError(f"tempo deve essere uno tra {config.FATTORI_TEMPO}")
    if not r.completato or not r.cartella or not (Path(r.cartella) / "note.json").exists():
        raise ValueError("Si può cambiare accordatura solo su una trascrizione completata.")
    nuovo = copy.deepcopy(r)
    out = Path(r.cartella)
    t0, inizio_ns = time.perf_counter(), time.time_ns()
    with ContextTracing().trace(f"riaccorda {accordatura}") as traccia:
        prima = bool(r.ripetizioni.get("attiva"))
        rip = prima if ripetizioni is None else ripetizioni
        sempl_prima = bool(r.semplifica.get("attiva"))
        sempl = sempl_prima if semplifica is None else semplifica
        fattore = r.fattore_tempo if tempo is None else tempo
        if fattore != r.fattore_tempo or not (out / "ritmo_tablatura.json").exists():
            _ritmo_tablatura(nuovo, out, fattore)
            nuovo.avvisi_per_passo.pop("tempo", None)          # l'avviso del dimezzamento automatico non vale più
        if ((stile is not None and stile != r.stile) or rip != prima or sempl != sempl_prima
                or fattore != r.fattore_tempo):
            _power_chord(nuovo, out, stile or r.stile)
            if rip:
                _ripetizioni(nuovo, out)
            else:
                nuovo.ripetizioni = {"attiva": False}
                nuovo.avvisi_per_passo.pop("ripetizioni", None)
                nuovo.decisioni = [d for d in nuovo.decisioni if not d.startswith("Confronto tra ripetizioni")]
            if sempl:
                _semplifica(nuovo, out)
            else:
                nuovo.semplifica = {"attiva": False}
                nuovo.avvisi_per_passo.pop("semplifica", None)
                nuovo.decisioni = [d for d in nuovo.decisioni if not d.startswith("Note ripetute")]
        _tablatura_e_verifica(nuovo, out, accordatura)
        nuovo.decisioni = [d for d in nuovo.decisioni if not d.startswith("Accordatura scelta")]
        nuovo.decisioni.append(f"Accordatura scelta dall'utente: {accordatura} (tablatura rifatta).")
        nuovo.avvisi = _unisci_avvisi(*nuovo.avvisi_per_passo.values())
        _scrivi_resoconto(nuovo, resoconto_llm)
        passi, t_in, t_out = _salva_traccia(traccia.get_spans(), out / "traccia_riaccorda.json", inizio_ns)
    nuovo.passi = r.passi + passi
    nuovo.token_in, nuovo.token_out = r.token_in + t_in, r.token_out + t_out
    nuovo.costo_usd = _costo(nuovo.token_in, nuovo.token_out)
    nuovo.tempo_totale_s = round(time.perf_counter() - t0, 2)
    return nuovo


# ---------------------------------------------------------------- modalità "agente"
PROMPT_AGENTE = (
    "Crei tablature di chitarra da file audio. Procedura: 1) valida_file_audio; se non valido fermati. "
    "2) separa_chitarra_audio; se la chitarra non è presente fermati. "
    "3) analizza_ritmo_audio sul file ORIGINALE. "
    "4) trascrivi_note_audio sull''audio_consigliato' passando il 'bpm'; se gli avvisi suggeriscono "
    "un'altra sensibilità riprova una sola volta. "
    "4b) completa_power_chord_audio con 'file_note', 'audio_basso' e 'audio_chitarra' (modalita 'auto'). "
    "5) crea_tablatura_audio con il 'file_note' restituito al passo 4b e 'file_ritmo'. "
    "6) verifica_tablatura_audio con il 'file_tablatura'. "
    "Alla fine rispondi in italiano in 2-3 frasi per un chitarrista, oppure 'STOP: <motivo>'."
)


def trascrivi_brano_agente(percorso: str | Path, cartella_output: str | Path | None = None) -> RisultatoBrano:
    """Stessa pipeline, ma è l'agente LLM a decidere la sequenza dei tool. Gli avvisi e i file
    sono comunque estratti dal codice, leggendo i risultati dei tool nella traccia."""
    from datapizza.agents import Agent

    from app.tools.power_chord import completa_power_chord_audio
    from app.tools.ritmo import analizza_ritmo_audio
    from app.tools.separazione import separa_chitarra_audio
    from app.tools.tablatura import crea_tablatura_audio
    from app.tools.trascrizione import trascrivi_note_audio
    from app.tools.validazione import valida_file_audio
    from app.tools.verifica import verifica_tablatura_audio

    client = _client_llm()
    if client is None:
        raise RuntimeError("La modalità agente richiede OPENAI_API_KEY nel file .env.")
    p = Path(percorso).resolve()
    out = Path(cartella_output) if cartella_output else Path(config.CARTELLA_OUTPUT) / p.stem
    r = RisultatoBrano(completato=False, modalita="agente")
    agente = Agent(name="orchestratore", client=client, system_prompt=PROMPT_AGENTE, max_steps=config.MAX_PASSI_AGENTE,
                   tools=[valida_file_audio, separa_chitarra_audio, analizza_ritmo_audio, trascrivi_note_audio,
                          completa_power_chord_audio, crea_tablatura_audio, verifica_tablatura_audio])
    t0, inizio_ns = time.perf_counter(), time.time_ns()
    with ContextTracing().trace(f"agente {p.name}") as traccia:
        try:
            risposta = agente.run(f"Crea la tablatura del file: {p}")
            r.resoconto = (risposta.text or "").strip() or None
        except Exception as e:
            r.motivo_stop = f"Errore dell'agente: {type(e).__name__}: {e}"
        spans = traccia.get_spans()
        # risultati dei tool dalla traccia (guardrail O27: gli avvisi non dipendono dall'LLM)
        # (si tiene l'ULTIMO risultato di ogni tool: se l'agente ha ripetuto la trascrizione,
        # l'avviso "riprovare con 'alta'" della prima non vale più)
        risultati: dict[str, dict] = {}
        chiamate: dict[str, int] = {}
        for s in sorted(spans, key=lambda s: s.start_time):
            a = dict(s.attributes or {})
            if a.get("type") == "tool" and "tool_result" in a:
                try:
                    dati = json.loads(a["tool_result"])
                except (TypeError, json.JSONDecodeError):
                    continue
                nome = s.name.removeprefix("Tool ")
                risultati[nome] = dati
                chiamate[nome] = chiamate.get(nome, 0) + 1
        if chiamate.get("trascrivi_note_audio", 0) > 1:
            r.decisioni.append(f"L'agente ha ripetuto la trascrizione con sensibilità "
                               f"{risultati['trascrivi_note_audio'].get('sensibilita')}.")
        r.avvisi = _unisci_avvisi(*[d.get("avvisi", []) + d.get("errori", []) for d in risultati.values()])
        r.passi, r.token_in, r.token_out = _salva_traccia(spans, out / "traccia.json", inizio_ns)
    ver = risultati.get("verifica_tablatura_audio")
    tab = risultati.get("crea_tablatura_audio", {})
    rit = risultati.get("analizza_ritmo_audio", {})
    tr = risultati.get("trascrivi_note_audio", {})
    if ver:
        r.completato = True
        r.file_gp5, r.file_tablatura, r.file_testo = ver["file_gp5"], ver["file_tablatura"], ver["file_testo"]
        r.suonabile, r.note_tolte_verifica, r.numero_note = ver["suonabile"], ver["note_tolte"], ver["numero_note"]
    elif not r.motivo_stop:
        r.motivo_stop = r.resoconto or "L'agente si è fermato prima della tablatura."
    r.accordatura, r.numero_battute = tab.get("accordatura"), tab.get("numero_battute")
    r.bpm, r.metro = rit.get("bpm"), rit.get("metro")
    r.tonalita = f"{rit['tonalita']} (relativa {rit.get('tonalita_relativa')})" if rit.get("tonalita") else None
    r.sensibilita, r.file_midi = tr.get("sensibilita"), tr.get("file_midi")
    r.file_traccia = str(out / "traccia.json")
    r.costo_usd = _costo(r.token_in, r.token_out)
    r.tempo_totale_s = round(time.perf_counter() - t0, 2)
    return r


def riassunto_testuale(r: RisultatoBrano) -> str:
    """Testo per il terminale (e base per la pagina web)."""
    if not r.completato:
        righe = [f"STOP: {r.motivo_stop}"]
    else:
        righe = [f"Tablatura: {r.file_gp5}",
                 f"Accordatura {r.accordatura} | {r.bpm} BPM | {r.metro} | tonalità {r.tonalita} | "
                 f"{r.numero_battute} battute, {r.numero_note} note | sensibilità {r.sensibilita}"]
        if r.resoconto:
            righe += ["", r.resoconto]
    if r.decisioni:
        righe += ["", "Decisioni:"] + [f"  - {d}" for d in r.decisioni]
    righe += ["", "Avvisi:"] + [f"  ! {a}" for a in r.avvisi]
    righe += ["", f"{r.tempo_totale_s} s | token {r.token_in} in + {r.token_out} out | ${r.costo_usd:.4f} | "
              f"traccia: {r.file_traccia}"]
    return "\n".join(righe)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(description="MP3 -> tablatura per chitarra")
    ap.add_argument("mp3")
    ap.add_argument("--accordatura", default="auto")
    ap.add_argument("--sensibilita", default="auto")
    ap.add_argument("--stile", default="auto", choices=["auto", "power_chord", "libero"])
    ap.add_argument("--separatore", default="demucs", choices=list(config.SEPARATORI))
    ap.add_argument("--ripetizioni", action="store_true", help="confronto tra battute ripetute (sperimentale)")
    ap.add_argument("--semplifica", action="store_true", help="unisce gli accordi riattaccati mentre suonano")
    ap.add_argument("--tempo", type=float, default=None, choices=list(config.FATTORI_TEMPO),
                    help="fattore del tempo (0.5 = metà); senza: automatico")
    ap.add_argument("--modalita", default="codice", choices=["codice", "agente"])
    ap.add_argument("--senza-llm", action="store_true", help="niente resoconto LLM (costo 0)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ris = trascrivi_brano(a.mp3, a.accordatura, a.sensibilita, resoconto_llm=not a.senza_llm, modalita=a.modalita,
                          stile=a.stile, separatore=a.separatore, ripetizioni=a.ripetizioni,
                          semplifica=a.semplifica, tempo=a.tempo)
    print(json.dumps(asdict(ris), ensure_ascii=False, indent=1) if a.json else riassunto_testuale(ris))
