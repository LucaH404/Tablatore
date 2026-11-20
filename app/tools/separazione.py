"""Tool 2 - separa_chitarra: isola la chitarra dal brano (Analista audio).

Usa Demucs `htdemucs_6s`, che divide il brano in 6 tracce
(batteria, basso, voce, chitarra, piano, altro). Oltre alla traccia di chitarra
restituisce misure che servono all'agente per decidere:
  - quota_chitarra: frazione dell'energia totale che appartiene alla chitarra
  - chitarra_presente: False se la quota è sotto la soglia (nel brano non c'è chitarra)
  - era_un_mix: False se il file è già solo chitarra (batteria+basso+voce+piano < 15%)
  - audio_consigliato: il file da passare alla trascrizione
    (l'originale se era già solo chitarra, così si evitano gli artefatti della separazione)
Non usa LLM: è un modello di separazione che gira in locale su GPU (o CPU).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datapizza.tools import tool  # noqa: E402

from app import config  # noqa: E402
from app.tools.audio_io import carica_audio, salva_mp3, salva_wav  # noqa: E402

_separatore = None  # il modello si carica una volta sola e resta in memoria
_roformer = None


def _get_separatore():
    global _separatore
    if _separatore is None:
        from demucs.api import Separator
        _separatore = Separator(config.MODELLO_DEMUCS, shifts=config.DEMUCS_SHIFTS)
    return _separatore


def _tracce_demucs(audio, sr) -> tuple[dict, str]:
    sep = _get_separatore()
    _, tracce = sep.separate_tensor(torch.from_numpy(audio), sr)
    return {k: v.cpu().numpy() for k, v in tracce.items()}, str(sep._device)


def _get_roformer():
    """BS-RoFormer SW (pacchetto bs-roformer-infer): pesi scaricati al primo uso (700 MB)."""
    global _roformer
    if _roformer is None:
        import yaml
        from bs_roformer import ensure_model_assets, get_model_from_config
        from bs_roformer.inference import SafeLoaderWithTuple
        from ml_collections import ConfigDict
        ckpt, cfg = ensure_model_assets(config.MODELLO_ROFORMER)
        with open(cfg) as f:
            conf = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
        modello = get_model_from_config("bs_roformer", conf)
        modello.load_state_dict(torch.load(ckpt, map_location="cpu"))
        dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _roformer = (modello.to(dispositivo).eval(), conf, dispositivo)
    return _roformer


def _tracce_roformer(audio) -> tuple[dict, str]:
    import contextlib
    import io
    import warnings

    from bs_roformer import demix_track
    modello, conf, dispositivo = _get_roformer()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():   # niente stampe di avanzamento
        warnings.simplefilter("ignore")
        tracce, _ = demix_track(conf, modello, torch.tensor(audio, dtype=torch.float32), dispositivo)
    return tracce, str(dispositivo)


@dataclass
class RisultatoSeparazione:
    audio_chitarra: str
    audio_consigliato: str
    audio_basso: str
    quota_chitarra: float
    chitarra_presente: bool
    era_un_mix: bool
    quote_energia: dict
    durata_s: float
    tempo_elaborazione_s: float
    dispositivo: str
    avvisi: list[str] = field(default_factory=list)
    separatore: str = "demucs"          # quello usato davvero (dopo l'eventuale ripiego)


def _quote(tracce: dict) -> dict:
    energie = {nome: float((np.asarray(t, dtype=np.float64) ** 2).sum()) for nome, t in tracce.items()}
    totale = sum(energie.values()) or 1.0
    return {nome: round(e / totale, 4) for nome, e in energie.items()}


def separa_chitarra(percorso: str | Path, cartella_output: str | Path | None = None,
                    separatore: str = "demucs") -> RisultatoSeparazione:
    """Isola la traccia di chitarra e misura quanto la chitarra pesa nel brano.
    'cartella_output' è la cartella finale dei file (come per gli altri tool); se manca,
    output/<nome del file>. 'separatore': "demucs" (default) o "roformer" (sperimentale: se
    non trova la chitarra o non si carica, si ripiega su Demucs con un avviso)."""
    if separatore not in config.SEPARATORI:
        raise ValueError(f"separatore deve essere uno tra {config.SEPARATORI}")
    p = Path(percorso)
    out_dir = Path(cartella_output) if cartella_output else Path(config.CARTELLA_OUTPUT) / p.stem
    sr = config.SAMPLE_RATE_LAVORO

    t0 = time.perf_counter()
    audio = carica_audio(p, sr, canali=2)
    avvisi = []
    usato = separatore
    if separatore == "roformer":
        try:
            tracce, dispositivo = _tracce_roformer(audio)
            quote = _quote(tracce)
            if quote["guitar"] < config.QUOTA_CHITARRA_MIN:
                avvisi.append(f"Il separatore RoFormer non ha trovato la chitarra ({quote['guitar']:.1%} "
                              "dell'energia): usato Demucs.")
                usato = "demucs"
        except Exception as e:                       # pesi non scaricabili, memoria...
            avvisi.append(f"Separatore RoFormer non disponibile ({type(e).__name__}): usato Demucs.")
            usato = "demucs"
    if usato == "demucs":
        tracce, dispositivo = _tracce_demucs(audio, sr)
        quote = _quote(tracce)
    quota = quote["guitar"]

    file_chitarra = salva_wav(out_dir / "chitarra.wav", tracce["guitar"], sr)
    try:            # l'utente ascolta quello che "sente" l'applicativo (richiesta dello Step 20)
        salva_mp3(file_chitarra, out_dir / "chitarra.mp3")
    except Exception as e:
        avvisi.append(f"MP3 della chitarra isolata non creato ({e}): la trascrizione prosegue.")
    file_originale = salva_wav(out_dir / "originale.wav", audio, sr)
    # il basso serve al Tool 4b: suona le fondamentali che la trascrizione della chitarra perde
    file_basso = salva_wav(out_dir / "basso.wav", tracce["bass"], sr)
    (out_dir / "linea_basso.npz").unlink(missing_ok=True)      # linea del basso di un'altra separazione
    dt = time.perf_counter() - t0

    accompagnamento = sum(quote[t] for t in config.TRACCE_ACCOMPAGNAMENTO)
    presente = quota >= config.QUOTA_CHITARRA_MIN
    # Da sola, la chitarra viene in parte assegnata alla traccia "other": per capire se il file
    # è un mix si guarda quanto pesano gli strumenti di accompagnamento, non la sola chitarra.
    mix = accompagnamento >= config.QUOTA_ACCOMPAGNAMENTO_MAX_SOLO
    if not presente:
        avvisi.append(f"Nel brano la chitarra è quasi assente ({quota:.1%} dell'energia): "
                      "la trascrizione non sarebbe affidabile.")
    elif mix and quota < 0.15:
        avvisi.append(f"La chitarra è poco presente nel mix ({quota:.1%}): la traccia isolata "
                      "può contenere residui di altri strumenti.")

    return RisultatoSeparazione(
        audio_chitarra=str(file_chitarra),
        audio_consigliato=str(file_chitarra if mix else file_originale),
        audio_basso=str(file_basso),
        quota_chitarra=quota,
        chitarra_presente=presente,
        era_un_mix=mix,
        quote_energia=quote,
        durata_s=round(audio.shape[1] / sr, 2),
        tempo_elaborazione_s=round(dt, 2),
        dispositivo=dispositivo,
        avvisi=avvisi,
        separatore=usato,
    )


@tool
def separa_chitarra_audio(percorso: str) -> str:
    """Isola la chitarra da un file audio già validato. Restituisce un JSON con:
    'audio_consigliato' (il file da usare per la trascrizione), 'chitarra_presente'
    (se false NON proseguire con la trascrizione e avvisare l'utente), 'era_un_mix',
    'quota_chitarra' (0-1, quanto pesa la chitarra nel brano), 'quote_energia' per strumento
    e 'avvisi' da mostrare all'utente."""
    return json.dumps(asdict(separa_chitarra(percorso)), ensure_ascii=False)
