"""Step 24: regola per comporre la traccia "Melodia" battuta per battuta (Dream On).

Per ogni battuta audio (allineamento DTW come valuta_dream_on.py) e per ogni parte trascritta
(sinistra / centro+destra / intera) calcola l'F1 delle altezze rispetto alla Lead Guitar di Songsterr
("oracolo") e alcuni indizi calcolabili senza riferimento: altezza media, numero di note, energia della
parte, similitudine delle note tra i due lati. Poi prova regole semplici e ne stampa il risultato
(F1 medio per sezione) contro l'oracolo e contro le parti fisse.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_melodia.py <lavoro> <cart_sinistra> <cart_centro_destra> <cart_intera>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import riferimento_nine_thou_completo as base  # noqa: E402
from riferimento_dream_on import CORDE, SEZIONI_BATTUTE, battute_suonate, riferimento_battute  # noqa: E402


def ritmo(d: Path) -> Path:
    return d / "ritmo_tablatura.json" if (d / "ritmo_tablatura.json").exists() else d / "ritmo.json"


def note_tab(d: Path) -> list[tuple[float, int]]:
    tab = json.loads((d / "tablatura_verificata.json").read_text(encoding="utf-8"))
    return [(e["tempo_s"], n["midi"]) for b in tab["battute"] for e in b["eventi"] if "tempo_s" in e for n in e["note"]]


def f1(vere: set, nostre: set) -> float:
    if not vere and not nostre:
        return 1.0
    ok = len(vere & nostre)
    return 2 * ok / (len(vere) + len(nostre)) if (vere or nostre) else 0.0


def energia(wav: Path, limiti) -> np.ndarray:
    y, sr = sf.read(wav)
    y = y.mean(1) if y.ndim > 1 else y
    return np.array([float(np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2) + 1e-12)) for a, b in zip(limiti[:-1], limiti[1:])])


def timbro(wav: Path, limiti) -> dict[str, np.ndarray]:
    """Per battuta: brillantezza (centroide spettrale, Hz), rumorosità (piattezza), attività (forza degli attacchi)."""
    import librosa
    y, sr = sf.read(wav)
    y = (y.mean(1) if y.ndim > 1 else y).astype(np.float32)
    hop = 1024
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    pia = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    att = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(cen)), sr=sr, hop_length=hop)
    out = {"centroide": [], "piattezza": [], "attivita": []}
    for a, b in zip(limiti[:-1], limiti[1:]):
        m = (t >= a) & (t < b)
        w = rms[m] + 1e-9                                   # media pesata dall'energia: i silenzi non contano
        out["centroide"].append(float(np.average(cen[m], weights=w)) if m.any() else 0.0)
        out["piattezza"].append(float(np.average(pia[m], weights=w)) if m.any() else 0.0)
        out["attivita"].append(float(att[m[: len(att)]].mean()) if m.any() else 0.0)
    return {k: np.array(v) for k, v in out.items()}


CLASSI_PANNS = {"distorsione": "Distortion", "elettrica": "Electric guitar", "acustica": "Acoustic guitar",
                "chitarra": "Guitar", "basso": "Bass guitar"}


def ispettore(wav: Path, limiti) -> dict[str, np.ndarray]:
    """Per battuta: probabilità media (PANNs, AudioSet) delle classi in CLASSI_PANNS. Cache in <wav>.panns.npz."""
    cache = wav.with_suffix(".panns.npz")
    if cache.exists():
        fw, fps = np.load(cache)["fw"], float(np.load(cache)["fps"])
    else:
        import librosa
        from panns_inference import SoundEventDetection, labels
        y, _ = librosa.load(str(wav), sr=32000, mono=True)
        sed = SoundEventDetection(checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"),
                                  device="cuda")
        tutte = sed.inference(y[None, :])[0]
        fw = tutte[:, [labels.index(v) for v in CLASSI_PANNS.values()]]
        fps = fw.shape[0] / (len(y) / 32000)
        np.savez(cache, fw=fw, fps=fps)
    out = {}
    for j, k in enumerate(CLASSI_PANNS):
        out[k] = np.array([float(fw[int(a * fps):max(int(a * fps) + 1, int(b * fps)), j].mean())
                           for a, b in zip(limiti[:-1], limiti[1:])])
    return out


def lisciata(scelte: list[str], k: int) -> list[str]:
    """Maggioranza su k battute vicine (k dispari)."""
    if k <= 1:
        return scelte
    h = k // 2
    return [max(set(fin := scelte[max(0, i - h): i + h + 1]), key=fin.count) for i in range(len(scelte))]


def main(lavoro: Path, sin: Path, cd: Path, intera: Path):
    mappa, limiti, _ = base.allinea(intera / "chitarra.wav", ritmo(intera), battute_suonate())
    rif = riferimento_battute()
    parti = {"sinistra": sin, "centro_destra": cd, "intera": intera}
    note = {k: note_tab(v) for k, v in parti.items()}
    en = {"sinistra": energia(lavoro / "stereo" / "chitarra_sinistra.wav", limiti),
          "centro_destra": energia(lavoro / "stereo" / "chitarra_centro_destra.wav", limiti)}
    tim = {"sinistra": timbro(lavoro / "stereo" / "chitarra_sinistra.wav", limiti),
           "centro_destra": timbro(lavoro / "stereo" / "chitarra_centro_destra.wav", limiti)}
    for k in tim:
        tim[k].update(ispettore(lavoro / "stereo" / f"chitarra_{k}.wav", limiti))
    righe = []
    for a, r in enumerate(mappa):
        b = r + 1
        vere = {CORDE[c] + f for c, f in rif[b]}
        if not vere:
            continue
        riga = {"battuta": b, "sezione": SEZIONI_BATTUTE[b]}
        for k in parti:
            alt = [m for t, m in note[k] if limiti[a] <= t < limiti[a + 1]]
            tempi = [t for t, m in note[k] if limiti[a] <= t < limiti[a + 1]]
            riga[k] = {"f1": f1(vere, set(alt)), "media": float(np.mean(alt)) if alt else 0.0,
                       "n": len(alt), "alt": set(alt),
                       "poli": len(alt) / max(1, len(set(round(t, 3) for t in tempi)))}   # note per attacco
        riga["e_sin"], riga["e_cd"] = en["sinistra"][a], en["centro_destra"][a]
        s, c = riga["sinistra"]["alt"], riga["centro_destra"]["alt"]
        riga["sim"] = len(s & c) / max(1, len(s | c))
        riga["proprie_sin"], riga["proprie_cd"] = len(s - c), len(c - s)
        for k, nome in (("sinistra", "sin"), ("centro_destra", "cd")):
            for f in ("centroide", "piattezza", "attivita") + tuple(CLASSI_PANNS):
                riga[f"{f}_{nome}"] = tim[k][f][a]
        righe.append(riga)

    regole = {
        "sempre sinistra": lambda r: "sinistra",
        "sempre centro+destra": lambda r: "centro_destra",
        "sempre intera": lambda r: "intera",
        "più acuta (sin/cd)": lambda r: "sinistra" if r["sinistra"]["media"] > r["centro_destra"]["media"] else "centro_destra",
        "più energia (sin/cd)": lambda r: "sinistra" if r["e_sin"] > r["e_cd"] else "centro_destra",
        "sin se più acuta di 3+ semitoni, altrimenti cd": lambda r: "sinistra" if r["sinistra"]["media"] > r["centro_destra"]["media"] + 3 else "centro_destra",
        "sin se più acuta e più forte, altrimenti cd": lambda r: "sinistra" if (r["sinistra"]["media"] > r["centro_destra"]["media"] and r["e_sin"] > r["e_cd"]) else "centro_destra",
        "accordi (intera ≥2 note/attacco) → intera, poi acuta+forte": lambda r: "intera" if r["intera"]["poli"] >= 2 else (
            "sinistra" if (r["sinistra"]["media"] > r["centro_destra"]["media"] and r["e_sin"] > r["e_cd"]) else "centro_destra"),
        "accordi (≥2,5) → intera, poi acuta+forte": lambda r: "intera" if r["intera"]["poli"] >= 2.5 else (
            "sinistra" if (r["sinistra"]["media"] > r["centro_destra"]["media"] and r["e_sin"] > r["e_cd"]) else "centro_destra"),
        "accordi (≥1,6) → intera, poi acuta+forte": lambda r: "intera" if r["intera"]["poli"] >= 1.6 else (
            "sinistra" if (r["sinistra"]["media"] > r["centro_destra"]["media"] and r["e_sin"] > r["e_cd"]) else "centro_destra"),
        "oracolo (sin/cd)": lambda r: max(("sinistra", "centro_destra"), key=lambda k: r[k]["f1"]),
        "oracolo (tutte)": lambda r: max(parti, key=lambda k: r[k]["f1"]),
    }
    sc = lambda c: "sinistra" if c else "centro_destra"
    regole.update({
        "più note proprie": lambda r: sc(r["proprie_sin"] > r["proprie_cd"]),
        "più brillante (centroide)": lambda r: sc(r["centroide_sin"] > r["centroide_cd"]),
        "più rumorosa (piattezza)": lambda r: sc(r["piattezza_sin"] > r["piattezza_cd"]),
        "più attiva (attacchi)": lambda r: sc(r["attivita_sin"] > r["attivita_cd"]),
        "più forte e più note proprie": lambda r: sc(r["e_sin"] > r["e_cd"] and r["proprie_sin"] > r["proprie_cd"]),
        "forte o (note proprie e acuta)": lambda r: sc(r["e_sin"] > r["e_cd"] or (r["proprie_sin"] > r["proprie_cd"] and r["sinistra"]["media"] > r["centro_destra"]["media"])),
        "energia relativa sin > 0,8 cd e acuta": lambda r: sc(r["e_sin"] > 0.8 * r["e_cd"] and r["sinistra"]["media"] > r["centro_destra"]["media"]),
        "Ispettore: più distorta": lambda r: sc(r["distorsione_sin"] > r["distorsione_cd"]),
        "Ispettore: più 'chitarra elettrica'": lambda r: sc(r["elettrica_sin"] > r["elettrica_cd"]),
        "Ispettore: meno 'basso'": lambda r: sc(r["basso_sin"] < r["basso_cd"]),
        "Ispettore: distorta e meno basso": lambda r: sc(r["distorsione_sin"] > r["distorsione_cd"] and r["basso_sin"] < r["basso_cd"]),
        "Ispettore: distorta di +0,05": lambda r: sc(r["distorsione_sin"] > r["distorsione_cd"] + 0.05),
    })
    sezioni = list(dict.fromkeys(r["sezione"] for r in righe))
    print(f"battute con note nel riferimento: {len(righe)}")
    print(f"{'regola':48s} {'liscia':>6s} {'tutto':>6s} " + " ".join(f"{s[:10]:>10s}" for s in sezioni))
    for nome, regola in regole.items():
        for k in ((1,) if nome.startswith(("oracolo", "sempre")) else (1, 3, 5)):
            scelte = lisciata([regola(r) for r in righe], k)
            per = defaultdict(list)
            for r, s in zip(righe, scelte):
                per[r["sezione"]].append(r[s]["f1"])
                per["tutto"].append(r[s]["f1"])
            print(f"{nome[:48]:48s} {k:6d} {100 * np.mean(per['tutto']):5.0f}% " + " ".join(f"{100 * np.mean(per[s]):9.0f}%" for s in sezioni))
    print("\nindizi medi per sezione (sin / cd): note proprie | centroide Hz | piattezza | attività | parte migliore (oracolo)")
    for s in sezioni:
        rr = [r for r in righe if r["sezione"] == s]
        mig = sum(1 for r in rr if r["sinistra"]["f1"] > r["centro_destra"]["f1"])
        m = lambda f: f"{np.mean([r[f + '_sin'] for r in rr]):.3g} / {np.mean([r[f + '_cd'] for r in rr]):.3g}"
        print(f"   {s:22s} {m('proprie')} | {m('centroide')} | {m('piattezza')} | {m('attivita')} | sinistra meglio in {mig}/{len(rr)}")
    print("\nIspettore per sezione (sin / cd): " + " | ".join(CLASSI_PANNS))
    for s in sezioni:
        rr = [r for r in righe if r["sezione"] == s]
        m = lambda f: f"{np.mean([r[f + '_sin'] for r in rr]):.2f}/{np.mean([r[f + '_cd'] for r in rr]):.2f}"
        print(f"   {s:22s} " + " | ".join(m(k) for k in CLASSI_PANNS))
    # indizi medi per sezione
    print("\nnote per attacco nella traccia intera, per sezione:",
          {s: round(float(np.mean([r["intera"]["poli"] for r in righe if r["sezione"] == s])), 2) for s in sezioni})
    print("\nindizi per sezione: altezza media sin / cd, energia sin / cd, similitudine note")
    for s in sezioni:
        rr = [r for r in righe if r["sezione"] == s]
        print(f"   {s:22s} {np.mean([r['sinistra']['media'] for r in rr]):5.1f} / {np.mean([r['centro_destra']['media'] for r in rr]):5.1f}"
              f"   {np.mean([r['e_sin'] for r in rr]):.3f} / {np.mean([r['e_cd'] for r in rr]):.3f}   {np.mean([r['sim'] for r in rr]):.2f}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
