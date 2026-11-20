"""Valuta il Verificatore di suonabilità (Tool 6).

1. Oracolo: tablature create dalle note ESATTE (brano di prova, "Nine thou" da Songsterr): il
   Verificatore non deve trovare errori né togliere note. Si misura anche la velocità massima
   della mano nelle tablature vere (per tarare VELOCITA_MANO_MAX).
2. Reale: trascrizione 'alta' -> tablatura -> verifica. Nella finestra con riferimento si conta
   quante note tolte erano false e quante vere (precisione prima/dopo).

Uso:  .venv\\Scripts\\python.exe scripts\\valuta_verifica.py [cartella_cache]
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from app.tools.ritmo import analizza_ritmo  # noqa: E402
from app.tools.separazione import separa_chitarra  # noqa: E402
from app.tools.tablatura import crea_tablatura  # noqa: E402
from app.tools.trascrizione import trascrivi_note  # noqa: E402
from app.tools.verifica import _eventi_con_note, _posizione, verifica_tablatura  # noqa: E402
from valuta_tablatura import confronta, note_tab, riferimenti_nine_thou, scrivi_note  # noqa: E402

AUDIO = ROOT / "tests" / "audio"
REALE = AUDIO / "reale"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())


def velocita_max(file_json):
    tab = json.loads(Path(file_json).read_text(encoding="utf-8"))
    ev = _eventi_con_note(tab)
    v = []
    for (_, a), (_, b) in zip(ev, ev[1:]):
        pa, pb = _posizione(a["note"]), _posizione(b["note"])
        if pa is not None and pb is not None and abs(pb - pa) >= 5:
            v.append((abs(pb - pa) / (b["tempo_s"] - a["tempo_s"]), abs(pb - pa), b["tempo_s"] - a["tempo_s"]))
    return max(v, default=(0, 0, 0))


print("=== 1. ORACOLO ===")
gt = json.loads((AUDIO / "gt_note.json").read_text())["note"]
rit = analizza_ritmo(AUDIO / "mix.mp3", CACHE / "mix")
r = crea_tablatura(scrivi_note(gt, "sintetico"), rit.file_ritmo, cartella_output=CACHE / "o_mix")
v = verifica_tablatura(r.file_tablatura)
print(f"brano di prova: problemi {v.problemi_iniziali}, tolte {v.note_tolte}, riditeggiate {v.note_riditeggiate}, "
      f"velocità mano max {velocita_max(r.file_tablatura)}")
file_reali = sorted(REALE.glob("*.mp3")) if REALE.exists() else []
for f in file_reali:
    rif = riferimenti_nine_thou(f.name)
    rit = analizza_ritmo(f, CACHE / f.stem)
    r = crea_tablatura(scrivi_note(rif, f.stem), rit.file_ritmo, cartella_output=CACHE / f"o_{f.stem}")
    v = verifica_tablatura(r.file_tablatura)
    print(f"{f.name}: problemi {v.problemi_iniziali} {[p['tipo'] for p in v.problemi]}, tolte {v.note_tolte}, "
          f"riditeggiate {v.note_riditeggiate}, velocità mano max (tasti/s, tasti, s) {velocita_max(r.file_tablatura)}")

print("\n=== 2. REALE (trascrizione 'alta') ===")
for f in file_reali:
    rif = riferimenti_nine_thou(f.name)
    d = CACHE / f.stem
    rit = analizza_ritmo(f, d)
    file_note = d / "note.json"
    if not file_note.exists():
        ch = separa_chitarra(f, d / "sep").audio_consigliato
        file_note = Path(trascrivi_note(ch, "alta", d, bpm=rit.bpm).file_note)
    t_min, t_max = min(n["inizio"] for n in rif) - 0.1, max(n["inizio"] for n in rif) + 0.1
    for acc in ("auto", "drop C#"):
        r = crea_tablatura(file_note, rit.file_ritmo, acc, d / f"r_{acc}")
        v = verifica_tablatura(r.file_tablatura)
        prima = [n for n in note_tab(r.file_tablatura)[1] if t_min <= n["tempo_s"] <= t_max]
        dopo = [n for n in note_tab(v.file_tablatura)[1] if t_min <= n["tempo_s"] <= t_max]
        rec0, prec0, _ = confronta(rif, prima)
        rec1, prec1, _ = confronta(rif, dopo)
        tipi = {}
        for p in v.problemi:
            tipi[p["tipo"]] = tipi.get(p["tipo"], 0) + 1
        print(f"{f.name} acc={r.accordatura}: problemi {v.problemi_iniziali} -> {v.problemi_finali} {tipi} in {v.cicli} cicli, "
              f"tolte {v.note_tolte}/{r.numero_note}, riditeggiate {v.note_riditeggiate}, tasto max {r.tasto_max} -> {v.tasto_max} | "
              f"finestra rif.: note {len(prima)} -> {len(dopo)}, richiamo {rec0:.0%} -> {rec1:.0%}, "
              f"precisione {prec0:.0%} -> {prec1:.0%} | {v.tempo_elaborazione_s}s")
        if acc == "auto":
            print(Path(v.file_testo).read_text(encoding="utf-8")[:1200])
