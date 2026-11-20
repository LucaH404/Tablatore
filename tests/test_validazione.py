"""Test del Tool 1 (valida_file). I file audio di prova vengono generati con ffmpeg."""
import json
import shutil
import subprocess

import pytest

from app.tools.validazione import valida_file, valida_file_audio

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg non installato")


def genera(path, secondi, formato_args):
    """Genera un tono di prova (La 440 Hz) della durata richiesta."""
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={secondi}",
           *formato_args, str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def mp3_valido(tmp_path):
    return genera(tmp_path / "brano.mp3", 10, ["-b:a", "128k"])


def test_mp3_valido_passa_con_avvisi_fissi(mp3_valido):
    r = valida_file(mp3_valido)
    assert r.valido, r.errori
    assert r.metadati["codec"] == "mp3"
    assert 9.5 < r.metadati["durata_s"] < 10.5
    assert any("copyright" in a for a in r.avvisi)
    assert any("approssimativa" in a for a in r.avvisi)


def test_file_inesistente(tmp_path):
    r = valida_file(tmp_path / "non_esiste.mp3")
    assert not r.valido and "non trovato" in r.errori[0]


def test_file_vuoto(tmp_path):
    p = tmp_path / "vuoto.mp3"
    p.write_bytes(b"")
    r = valida_file(p)
    assert not r.valido and "vuoto" in r.errori[0]


def test_testo_rinominato_in_mp3(tmp_path):
    p = tmp_path / "finto.mp3"
    p.write_text("non sono un file audio " * 100)
    r = valida_file(p)
    assert not r.valido
    assert any("audio leggibile" in e for e in r.errori)


def test_wav_rinominato_in_mp3(tmp_path):
    p = genera(tmp_path / "vero.wav", 5, [])
    finto = p.rename(tmp_path / "finto.mp3")
    r = valida_file(finto)
    assert not r.valido
    assert any("contiene audio" in e for e in r.errori)


def test_estensione_non_ammessa(tmp_path):
    r = valida_file(genera(tmp_path / "brano.wav", 5, []))
    assert not r.valido
    assert any("Formato non ammesso" in e for e in r.errori)


def test_troppo_corto(tmp_path):
    r = valida_file(genera(tmp_path / "corto.mp3", 1, ["-b:a", "128k"]))
    assert not r.valido
    assert any("troppo corto" in e for e in r.errori)


def test_troppo_lungo(tmp_path):
    r = valida_file(genera(tmp_path / "lungo.mp3", 7 * 60, ["-b:a", "32k"]))
    assert not r.valido
    assert any("troppo lungo" in e for e in r.errori)


def test_troppo_grande(tmp_path):
    p = tmp_path / "enorme.mp3"
    p.write_bytes(b"\0" * (21 * 1_048_576))
    r = valida_file(p)
    assert not r.valido
    assert any("troppo grande" in e for e in r.errori)


def test_bitrate_basso_passa_con_avviso(tmp_path):
    r = valida_file(genera(tmp_path / "lofi.mp3", 10, ["-b:a", "64k"]))
    assert r.valido, r.errori
    assert any("Bitrate basso" in a for a in r.avvisi)


def genera_sequenza(path, tratti):
    """MP3 fatto di tratti ('s', secondi) = silenzio e ('t', secondi) = tono, in sequenza."""
    ingressi, n = [], len(tratti)
    for tipo, sec in tratti:
        sorgente = f"anullsrc=r=44100:cl=mono:d={sec}" if tipo == "s" else f"sine=frequency=440:duration={sec}:sample_rate=44100"
        ingressi += ["-f", "lavfi", "-i", sorgente]
    subprocess.run([FFMPEG, "-y", "-v", "error", *ingressi, "-filter_complex",
                    "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1",
                    "-b:a", "128k", str(path)], check=True)
    return path


def test_silenzio_iniziale_e_finale(tmp_path):
    r = valida_file(genera_sequenza(tmp_path / "rec.mp3", [("s", 4), ("t", 8), ("s", 3)]))
    assert r.valido, r.errori
    assert 3.5 < r.metadati["silenzio_iniziale_s"] < 4.5
    assert 2.5 < r.metadati["silenzio_finale_s"] < 3.5
    assert 7.5 < r.metadati["durata_musica_s"] < 8.6
    assert any("all'inizio" in a for a in r.avvisi) and any("alla fine" in a for a in r.avvisi)


def test_pausa_a_meta(tmp_path):
    r = valida_file(genera_sequenza(tmp_path / "pausa.mp3", [("t", 5), ("s", 3), ("t", 5)]))
    assert r.valido
    assert any("a metà brano" in a for a in r.avvisi)
    assert r.metadati["silenzio_iniziale_s"] == 0 and r.metadati["silenzio_finale_s"] == 0


def test_file_muto_rifiutato(tmp_path):
    r = valida_file(genera_sequenza(tmp_path / "muto.mp3", [("s", 10)]))
    assert not r.valido
    assert any("silenzioso" in e for e in r.errori)


def test_nessun_silenzio_nessun_avviso(mp3_valido):
    r = valida_file(mp3_valido)
    assert not any("silenzio" in a.lower() for a in r.avvisi)
    assert r.metadati["silenzio_iniziale_s"] == 0


def test_tool_datapizza_restituisce_json(mp3_valido):
    # Il wrapper @tool espone lo schema che l'LLM vede e restituisce JSON
    assert valida_file_audio.name == "valida_file_audio"
    assert "percorso" in valida_file_audio.schema["parameters"]["properties"]
    dati = json.loads(valida_file_audio(str(mp3_valido)))
    assert dati["valido"] is True
