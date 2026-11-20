"""Esperimento: consenso tra ripetizioni (battute simili votano le note), misurato sul brano
intero di Nine Thou con il riferimento Songsterr. Non tocca la pipeline.

Uso: .venv\\Scripts\\python.exe scripts\\esperimento_ripetizioni.py output\\_banco\\nine_thou
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from riferimento_nine_thou_completo import allinea, battute_suonate, riferimento_temporale  # noqa: E402
from valuta_nine_thou_completo import valuta  # noqa: E402

SLOT = 4  # suddivisioni per beat per l'impronta della battuta


def griglie(ritmo):
    beat, down = np.array(ritmo["beat"]), np.array(ritmo["downbeat"])
    limiti = list(down) + [down[-1] + float(np.median(np.diff(down)))]
    out = []
    for a, b in zip(limiti[:-1], limiti[1:]):
        dentro = beat[(beat >= a - 0.05) & (beat < b - 0.05)]
        g = list(dentro[:4]) if len(dentro) >= 4 else list(np.linspace(a, b, 5)[:4])
        out.append(np.array(g + [b]))
    return out


def impronte(file_audio, gr, fmin="C2"):
    import librosa
    import soundfile as sf
    y, sr = sf.read(file_audio)
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    sr, hop = 22050, 256
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz(fmin))
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    t = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    F, E = [], []
    for g in gr:
        pts = np.concatenate([np.linspace(g[k], g[k + 1], SLOT + 1)[:-1] for k in range(4)] + [[g[-1]]])
        cols, ons = [], []
        for a, b in zip(pts[:-1], pts[1:]):
            m = (t >= a) & (t < b)
            cols.append(ch[:, m].mean(axis=1) if m.any() else np.zeros(12))
            ons.append(on[m].max() if m.any() else 0)
        F.append((np.array(cols).T, np.array(ons)))
        m = (t >= g[0]) & (t < g[-1])
        E.append(20 * np.log10(rms[m].mean() + 1e-9))
    return F, np.array(E)


def similarita(F):
    X = []
    for ch, on in F:
        v = ch.flatten()
        v = v - v.mean()
        X.append(v / (np.linalg.norm(v) + 1e-9))
    X = np.array(X)
    return X @ X.T


def note_in_battute(note, gr):
    inizi = np.array([g[0] for g in gr] + [gr[-1][-1]])
    per = [[] for _ in gr]
    for n in note:
        a = int(np.searchsorted(inizi, n["inizio"] + 0.035) - 1)
        if 0 <= a < len(gr):
            g = gr[a]
            k = int(np.clip(np.searchsorted(g, n["inizio"] + 0.035) - 1, 0, 3))
            pos = k + (n["inizio"] - g[k]) / (g[k + 1] - g[k])
            per[a].append(dict(n, pos=float(pos)))
    return per


def tempo_da_pos(g, pos):
    k = int(np.clip(np.floor(pos), 0, 3))
    return float(g[k] + (pos - k) * (g[k + 1] - g[k]))


def consenso(per, gr, S, soglia_sim=0.8, min_gruppo=3, q_add=0.5, q_del=0.25, tol=0.15, k_max=12,
             togli=True, aggiungi=True):
    out = []
    n_add = n_del = 0
    for a in range(len(per)):
        vicini = [b for b in np.argsort(-S[a]) if b != a and S[a, b] >= soglia_sim][:k_max]
        gruppo = [a] + list(vicini)
        if len(gruppo) < min_gruppo:
            out += per[a]
            continue
        # supporto di ogni (altezza, posizione): in quante battute del gruppo c'è
        def supporto(midi, pos):
            return sum(any(m["midi"] == midi and abs(m["pos"] - pos) <= tol for m in per[b]) for b in gruppo) / len(gruppo)
        tenute = []
        for n in per[a]:
            if togli and supporto(n["midi"], n["pos"]) < q_del:
                n_del += 1
                continue
            tenute.append(n)
        if aggiungi:
            cand = [(m["midi"], m["pos"], m) for b in vicini for m in per[b]]
            visti = []
            for midi, pos, m in cand:
                if any(n["midi"] == midi and abs(n["pos"] - pos) <= tol for n in tenute + visti):
                    continue
                if supporto(midi, pos) >= q_add:
                    # posizione mediana tra le battute che la contengono
                    poss = [x["pos"] for b in vicini for x in per[b] if x["midi"] == midi and abs(x["pos"] - pos) <= tol]
                    p = float(np.median(poss))
                    durs = [x["fine"] - x["inizio"] for b in vicini for x in per[b] if x["midi"] == midi and abs(x["pos"] - pos) <= tol]
                    t0 = tempo_da_pos(gr[a], p)
                    nuova = {"inizio": round(t0, 4), "fine": round(t0 + float(np.median(durs)), 4), "midi": midi,
                             "intensita": m.get("intensita", 0.5), "pos": p, "consenso": True}
                    visti.append(nuova)
                    n_add += 1
            tenute += visti
        out += tenute
    return out, n_add, n_del


def purezza(S, mappa, soglia, k_max=12):
    bs = battute_suonate()
    dis = [bs[r]["disegno"] for r in mappa]
    coppie = giuste = con_gruppo = 0
    for a in range(len(dis)):
        v = [b for b in np.argsort(-S[a]) if b != a and S[a, b] >= soglia][:k_max]
        coppie += len(v)
        giuste += sum(dis[b] == dis[a] for b in v)
        con_gruppo += len(v) >= 2
    return giuste / max(coppie, 1), con_gruppo / len(dis), coppie / len(dis)


def metà(note, rif, confini, t_mezzo):
    r = {}
    for nome, sel in (("1a metà", lambda t: t < t_mezzo), ("2a metà", lambda t: t >= t_mezzo)):
        v = valuta([n for n in rif if sel(n["inizio"])], [n for n in note if sel(n["inizio"])], confini)
        r[nome] = (round(v["richiamo"], 3), round(v["precisione"], 3))
    return r


if __name__ == "__main__":
    d = Path(sys.argv[1])
    ritmo = json.loads((d / "ritmo.json").read_text(encoding="utf-8"))
    gr = griglie(ritmo)
    note = json.loads((d / "note_power_chord.json").read_text(encoding="utf-8"))["note"]
    rif, confini = riferimento_temporale(d / "chitarra.wav", d / "ritmo.json")
    mappa, _, _ = allinea(d / "chitarra.wav", d / "ritmo.json")
    Fg, E = impronte(d / "chitarra.wav", gr)
    Fb, _ = impronte(d / "basso.wav", gr, fmin="C1")
    Sg = similarita(Fg)
    Sb = similarita(Fb)
    t_mezzo = gr[len(gr) // 2][0]
    for nome, S in (("chitarra", Sg), ("basso", Sb), ("chit+basso", (Sg + Sb) / 2)):
        for s in (0.6, 0.7, 0.8, 0.9):
            p, c, k = purezza(S, mappa, s)
            print(f"{nome:10s} soglia {s}: vicini dello stesso disegno {p:.0%}, battute con gruppo {c:.0%}, vicini medi {k:.1f}")
    v0 = valuta(rif, note, confini)
    print(f"\nbase: richiamo {v0['richiamo']:.1%} precisione {v0['precisione']:.1%} esatto {v0['attacco_esatto_richiamo']:.1%}"
          f" {metà(note, rif, confini, t_mezzo)}")
    per = note_in_battute(note, gr)
    # tetto: gruppi perfetti (stesso disegno nel riferimento)
    bs = battute_suonate()
    dis = np.array([bs[r]["disegno"] for r in mappa])
    So = (dis[:, None] == dis[None, :]).astype(float)
    for kw in [dict(soglia_sim=0.5), dict(soglia_sim=0.5, togli=False), dict(soglia_sim=0.5, q_add=0.34),
               dict(soglia_sim=0.5, k_max=40)]:
        nuove, na, nd = consenso(per, gr, So, **kw)
        v = valuta(rif, nuove, confini)
        print(f"ORACOLO {str(kw):40s} +{na:4d} -{nd:4d}: richiamo {v['richiamo']:.1%} precisione {v['precisione']:.1%} "
              f"esatto {v['attacco_esatto_richiamo']:.1%}")
    # errori sistematici: per ogni nota persa, in quante altre ripetizioni dello stesso disegno è stata trovata?
    from valuta_nine_thou_completo import assegna_beat
    est_b = {}
    for n in assegna_beat([dict(n) for n in note], confini):
        est_b.setdefault(n["beat"], set()).add(n["midi"])
    per_nota = {}
    for n in rif:
        if n["ghost"]:
            continue
        slot = round((n["inizio"] - confini[4 * n["battuta_audio"]]) / (confini[4 * n["battuta_audio"] + 4] - confini[4 * n["battuta_audio"]]) * 16)
        per_nota.setdefault((n["disegno"], slot, n["midi"]), []).append(n["midi"] in est_b.get(n["beat_globale"], set()))
    sempre = sum(len(v) for v in per_nota.values() if len(v) >= 3 and not any(v))
    a_volte = sum(v.count(False) for v in per_nota.values() if len(v) >= 3 and any(v))
    print(f"note perse in disegni ripetuti ≥3 volte: mai trovate in nessuna ripetizione {sempre}, "
          f"perse ma trovate altrove {a_volte}")
    # impronta dalle note trascritte (classe di altezza x 16 slot)
    X = []
    for ns in per:
        m = np.zeros((12, 16))
        for n in ns:
            m[n["midi"] % 12, min(int(round(n["pos"] * 4)), 15)] += 1
        v = m.flatten() - m.mean()
        X.append(v / (np.linalg.norm(v) + 1e-9))
    Sn = np.array(X) @ np.array(X).T
    for nome, S in (("note", Sn), ("note+audio", (Sg + Sb + Sn) / 3)):
        for s in (0.5, 0.6, 0.7, 0.8):
            p, c, k = purezza(S, mappa, s)
            print(f"{nome:10s} soglia {s}: vicini dello stesso disegno {p:.0%}, battute con gruppo {c:.0%}, vicini medi {k:.1f}")
            nuove, na, nd = consenso(per, gr, S, soglia_sim=s)
            v = valuta(rif, nuove, confini)
            print(f"      +{na:4d} -{nd:4d}: richiamo {v['richiamo']:.1%} precisione {v['precisione']:.1%} "
                  f"esatto {v['attacco_esatto_richiamo']:.1%} {metà(nuove, rif, confini, t_mezzo)}")
    S = (Sg + Sb) / 2
    for kw in [dict(), dict(togli=False), dict(aggiungi=False), dict(soglia_sim=0.7), dict(soglia_sim=0.9),
               dict(q_add=0.6), dict(q_add=0.4), dict(q_del=0.34), dict(q_del=0.15), dict(tol=0.1)]:
        nuove, na, nd = consenso(per, gr, S, **kw)
        v = valuta(rif, nuove, confini)
        print(f"{str(kw):28s} +{na:4d} -{nd:4d}: richiamo {v['richiamo']:.1%} precisione {v['precisione']:.1%} "
              f"esatto {v['attacco_esatto_richiamo']:.1%} {metà(nuove, rif, confini, t_mezzo)}")
