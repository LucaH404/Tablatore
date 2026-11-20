# Tablatore AI — da un MP3 alla tablatura per chitarra

Progetto personale costruito con il framework **datapizza-ai 0.1.0**.

Carichi un MP3, l'app isola la chitarra, ne trascrive le note e il ritmo e restituisce una **tablatura suonabile**:
la vedi e la ascolti nel browser, la scarichi in Guitar Pro, PDF, PNG, testo e MIDI. Un **tutor di teoria** (RAG)
risponde a domande sulla chitarra e sul brano aperto, e un **server MCP** rende gli stessi strumenti usabili da un
assistente AI come Claude Desktop.

> La trascrizione è automatica e approssimativa: va sempre verificata a orecchio.
> Le tablature di brani protetti da copyright sono solo per uso personale e di studio.

---

## Indice
1. [Cosa fa](#cosa-fa)
2. [Installazione](#installazione)
3. [Avvio](#avvio)
4. [Come funziona](#come-funziona)
5. [Componenti GenAI](#componenti-genai)
6. [Valutazione: qualità, costi, tempi](#valutazione-qualità-costi-tempi)
7. [Limiti noti](#limiti-noti)
8. [Struttura del progetto](#struttura-del-progetto)
9. [Licenze e crediti](#licenze-e-crediti)

---

## Cosa fa

- **Pagina web locale** (http://127.0.0.1:8000): carichi l'MP3, segui l'avanzamento passo per passo, poi
  - vedi la tablatura (e, se vuoi, il pentagramma) disegnata con alphaTab e la **ascolti** con un cursore che segue
    le note, con suono di chitarra acustica o elettrica distorta e velocità 50/75/100%;
  - leggi il **resoconto** scritto dall'IA, gli **avvisi** (dove fidarsi meno) e le **decisioni automatiche**;
  - cambi accordatura, stile, tempo (metà/doppio), semplificazione e premi **"Rifai la tablatura"**
    (pochi decimi di secondo: non ritrascrive);
  - scarichi `.gp5` (Guitar Pro / TuxGuitar), testo, MIDI, PDF, PNG, la traccia dei passi e la **chitarra isolata
    in MP3** (per sentire cosa ha "ascoltato" l'app);
  - chiedi al **tutor** ("che scala uso per improvvisare su questo brano?", "come si fa il palm mute?").
- **Riga di comando**: `python -m app.orchestratore brano.mp3`.
- **Server MCP**: `trascrivi_brano`, `rifai_tablatura`, `leggi_tablatura`, `chiedi_al_tutor`, `elenca_trascrizioni`.

---

## Installazione

Requisiti: Windows 10/11, macOS (Apple Silicon) o Linux; connessione internet; ~6 GB liberi con GPU NVIDIA
(~2 GB senza). Una GPU NVIDIA non è obbligatoria ma rende la separazione ~10 volte più veloce.
La chiave OpenAI è **facoltativa**: senza, la tablatura funziona lo stesso (mancano il resoconto e il tutor).

### Con lo script (consigliato)

**Windows** (PowerShell, nella cartella del progetto):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\inizializza.ps1
```
**macOS / Linux**:
```bash
bash scripts/inizializza.sh
```

Lo script, rilanciabile senza danni:
1. trova un Python di sistema (3.8+) e installa **uv**;
2. crea `.venv` con **Python 3.11** (uv lo scarica se manca);
3. installa **PyTorch 2.14.1** (CUDA 12.6 se trova una GPU NVIDIA, altrimenti CPU);
4. installa senza dipendenze Basic Pitch, beat_this e torchaudio (vedi sotto perché);
5. installa `requirements.txt` (versioni fissate);
6. controlla **ffmpeg** e propone di installarlo (winget / Homebrew / apt);
7. crea `.env` chiedendo la **tua** chiave OpenAI (input nascosto, mai stampata);
8. scarica in anticipo i modelli (Demucs, beat_this), il banco di suoni dell'anteprima e crea l'indice del tutor;
9. verifica l'ambiente (`scripts/check_setup.py`) e lancia i test (`pytest`).

Opzioni: `-Cpu` / `--cpu` (PyTorch per CPU anche con GPU), `-Sperimentali` / `--sperimentali` (separatore
BS-RoFormer, +700 MB), `-SenzaModelli`, `-SenzaTest`.

Provato su Windows 11 (installazione da zero in una cartella pulita, versione CPU: pronto in ~4 minuti con le cache
già presenti; con GPU e download completi contare 15-30 minuti). Lo script macOS/Linux è controllato solo nella
sintassi.

### A mano (riassunto)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # o /whl/cpu
uv pip install --python .venv --no-deps basic-pitch==0.4.0 beat-this==1.1.0 torchaudio==2.11.1 rotary-embedding-torch==0.9.2
uv pip install --python .venv -r requirements.txt
cp .env.esempio .env      # poi inserisci OPENAI_API_KEY
```
Serve **ffmpeg** nel PATH.

Perché alcune scelte:
- `basic-pitch` **senza dipendenze**: con le dipendenze installerebbe TensorFlow, in conflitto con numpy 2; l'app
  usa il suo modello ONNX con onnxruntime.
- `torchaudio`/`beat-this` **senza dipendenze**: altrimenti potrebbero sostituire la versione di PyTorch.
- `mcp==1.30.0` **fissato**: datapizza-ai accetta mcp 2.x, con cui il suo client MCP non funziona.
- **Niente Docker**: il vector store del tutor è Qdrant in modalità locale su disco (`dati/indice_tutor`).

---

## Avvio

| Cosa | Comando (Windows; su macOS/Linux `.venv/bin/python`) |
|---|---|
| Pagina web | `.venv\Scripts\python.exe -m app.web.server` → http://127.0.0.1:8000 |
| Riga di comando | `.venv\Scripts\python.exe -m app.orchestratore brano.mp3 [--accordatura "drop D"] [--senza-llm] [--json]` |
| Tutor da terminale | `.venv\Scripts\python.exe -m app.tutor "Cos'è un power chord?"` |
| Server MCP | `.venv\Scripts\python.exe -m app.mcp_server` (lo avvia il client MCP) |
| Test | `.venv\Scripts\python.exe -m pytest -q` |
| Valutazione del tutor | `.venv\Scripts\python.exe scripts\valuta_tutor.py --risposte` (~$0,01) |

Il server web ascolta solo su `127.0.0.1` (non è raggiungibile da altri computer).

### Server MCP in Claude Desktop

Aggiungi in `%APPDATA%\Claude\claude_desktop_config.json` (Windows) o
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), con il percorso del progetto:
```json
{
  "mcpServers": {
    "tablatore-ai": {
      "command": "C:\\percorso\\del\\progetto\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "env": { "PYTHONPATH": "C:\\percorso\\del\\progetto", "PYTHONIOENCODING": "utf-8" }
    }
  }
}
```
Poi, in Claude: *"Trascrivi C:\musica\brano.mp3, dimmi l'accordatura e chiedi al tutor che scala usare"*.
Il server è stato collaudato con il client MCP di datapizza-ai e con un agente datapizza (`scripts/prova_mcp.py`).

---

## Come funziona

### Agenti e tool

| Ruolo | Cosa fa | Implementazione |
|---|---|---|
| **Orchestratore** | coordina i passi, prende le decisioni con regola chiara, raccoglie avvisi e tracce | `app/orchestratore.py` (modalità "codice"; modalità "agente" datapizza per confronto) |
| **Guardiano dell'input** | MP3 valido, ≤ 20 MB, ≤ 6 minuti, non muto | `app/tools/validazione.py` (ffprobe, ffmpeg) |
| **Analista audio** | separa la chitarra, rileva ritmo e tonalità, trascrive le note, ricostruisce power chord e ottave | `separazione.py` (Demucs `htdemucs_6s`), `ritmo.py` (beat_this + librosa), `trascrizione.py` (Basic Pitch), `power_chord.py`, `ottave.py`, `ripetizioni.py`, `semplifica.py` |
| **Tablatore** | griglia ritmica, scelta dell'accordatura, corda e tasto di ogni nota | `tablatura.py` (programmazione dinamica), PyGuitarPro per il `.gp5` |
| **Verificatore di suonabilità** | estensione della mano, velocità dei salti, note impossibili o false | `verifica.py` |
| **Tutor di teoria** | RAG con memoria su 15 schede (teoria, tecnica, uso dell'app) | `app/tutor.py`, schede in `app/rag/corpus/` |

Principio guida: **i passi numerici sono codice deterministico** (modelli locali, nessun LLM); l'LLM serve solo dove
deve scrivere o spiegare (resoconto, tutor). Misurato: la modalità "agente", in cui un agente datapizza sceglie da
solo la sequenza dei tool, produce la stessa tablatura ma costa **~13 volte di più** e impiega 3-4 volte il tempo.

### La pipeline
1. **Validazione** → 2. **Separazione** (chitarra e basso isolati, chitarra salvata anche in MP3) →
3. **Ritmo** sul brano **originale** (battiti, battute, BPM, tonalità; tempo dimezzato se senza batteria e > 130 BPM) →
4. **Trascrizione** (sensibilità media, poi alta se < 1 nota per battito) →
4b **Power chord** dal basso o dalla chitarra (brani distorti) / **ottave** corrette (chitarra pulita) →
4c **Ripetizioni** e 4d **Semplifica** (opzioni) → 5. **Tablatura** → 6. **Verifica** → 7. **Resoconto** LLM.

### Il tutor (RAG)
- **Ingestione** (una volta): `MDParser → RecursiveSplitter → ChunkEmbedder (text-embedding-3-small)` → Qdrant
  locale; si rifà da sola se le schede cambiano.
- **Domanda**: guardrail → ricerca dei 4 passaggi più simili → prompt con memoria della conversazione, dati del
  brano aperto e passaggi con la fonte → `gpt-5.4-nano` → controllo delle fonti citate.
- La **scala consigliata** per il brano (nota, prima forma, tasti, accordature drop) è calcolata dal codice e
  passata al tutor: il modello piccolo non sapeva trasporre da solo la forma di esempio.

### Guardrail
- **Ingresso**: Guardiano dell'input (formato, dimensione, durata, silenzio); nella pagina limite di dimensione
  controllato durante il caricamento, nomi di file ripuliti, id casuali, download solo da un elenco fisso; nel
  tutor domande vuote/troppo lunghe, tentativi di cambiare le istruzioni e **domande fuori tema rifiutate senza
  chiamare l'LLM** (similitudine con le schede sotto soglia).
- **Uscita**: il **resoconto** viene scartato se contiene numeri che non sono nei dati (sostituito da un testo
  scritto dal codice); le risposte del **tutor** vengono scartate se citano schede non recuperate; il
  **Verificatore** garantisce una tablatura fisicamente suonabile; gli **avvisi** arrivano all'utente dal codice,
  non dall'LLM; due avvisi fissi (approssimazione, copyright) sempre visibili.

### Osservabilità
Ogni passo è uno span di **ContextTracing** (datapizza-ai / OpenTelemetry) con durata, decisioni, avvisi, errori e
token; la traccia è salvata in `traccia.json` (scaricabile dalla pagina). Il tutor traccia ricerca e generazione
(fonti, similitudine, esito, token). Costo in dollari calcolato per brano e per risposta.

---

## Componenti GenAI

| Componente | Dove |
|---|---|
| LLM e prompt (datapizza-ai) | resoconto (`orchestratore.py`), tutor (`tutor.py`), modalità agente; prompt di sistema per ruolo |
| Conoscenza documentale (RAG) | tutor: 15 schede, Qdrant, memoria della conversazione, contesto del brano, citazioni verificate |
| Autonomia e agenti | orchestratore con tracing, agente datapizza, server MCP usabile da altri host (es. Claude Desktop) |
| Sicurezza e osservabilità | guardrail in ingresso e in uscita, avvisi raccolti dal codice, tracce di ogni passo con token e costi |
| Valutazione con dati | qualità sui brani di prova, metriche del tutor, costi e tempi (sezione seguente) |
| Interfaccia e test | pagina web FastAPI + alphaTab, 132 test automatici, script di inizializzazione |

---

## Valutazione: qualità, costi, tempi

La qualità è misurata come la vede un chitarrista: confrontando **i tasti della tablatura** con tablature di
riferimento di Songsterr (fornite come screenshot dall'utente, mai scaricate in automatico), battuta per battuta,
**sul brano intero**.

| Brano di prova | Tipo | Risultato |
|---|---|---|
| *Lying From You* (versione strumentale di studio) | una chitarra distorta, riff e power chord | **90%** di tasti giusti per battito (**92,7%** con "Ripetizioni") |
| *Nine Thou* (registrazione con la band, 4 min) | chitarra distorta + batteria, basso, effetti | **63%** (Demucs) / **70,5%** (separatore sperimentale RoFormer) |
| *Dream On* (versione strumentale) | due chitarre (arpeggio + solista) | note: richiamo 74%, precisione 65%; tasti nell'arpeggio 79-85%, negli assoli ~12% |

**Tutor** (25 domande di prova, `tests/tutor_gold.json`): recupero della scheda giusta al primo posto **90%**, tra
le prime 4 **100%**, MRR **0,95**; 20/20 risposte citano la scheda attesa; 5/5 domande fuori tema rifiutate senza LLM.

| Operazione | Tempo | Costo |
|---|---|---|
| Brano di 30 s (GPU RTX 4070) | 5-6 s (14 s dalla pagina, resoconto compreso) | ≈ $0,0003 (resoconto) |
| "Rifai la tablatura" | 0,1-5 s | 0 |
| Domanda al tutor | 2,7 s mediani | ≈ $0,0006 |
| Indicizzazione del tutor (una volta) | ~6 s | ≈ $0,0001 |
| Trascrizione via MCP (30 s di audio) | 8,5 s | 0 |

Modelli: `gpt-5.4-nano` ($0,20 / $1,25 per milione di token in/out) e `text-embedding-3-small` ($0,02).

---

## Limiti noti

- **Assoli**: bending, vibrato, slide e legato veloce producono note in più o mancanti.
- **Due chitarre diverse** nello stesso brano vengono trascritte insieme. L'app sa **riconoscerle** (divisione
  per posizione nello stereo: 34,6% di tratti diversi su *Dream On* contro 0% sugli altri brani), ma comporre in
  automatico una traccia "melodia" non è risultato migliore della traccia intera: limite misurato e documentato.
- **Errori d'ottava** e diteggiature diverse da quelle originali (la stessa nota sta in più punti del manico).
- **Synth, scratch ed effetti** possono essere scambiati per chitarra.
- La soglia "fuori tema" del tutor è tarata su poche domande (margine 0,30 / 0,37).
- Il tutor usa un modello piccolo: risposte corrette nelle prove ma a volte prolisse.
- Pagina: alphaTab e i suoi font arrivano da CDN, quindi serve internet anche in locale.
- Lo script macOS/Linux non è stato eseguito su quei sistemi; Mac Intel non supportati (Demucs).

---

## Struttura del progetto

```
app/
  config.py            parametri (soglie, accordature, modelli, prezzi)
  orchestratore.py     pipeline, tracing, guardrail, CLI
  tutor.py             tutor di teoria (RAG)
  mcp_server.py        server MCP
  tools/               validazione, separazione, ritmo, trascrizione, power chord, ottave,
                       ripetizioni, semplifica, tablatura, verifica, audio_io
  rag/corpus/          15 schede del tutor (teoria/, tecnica/, app/)
  web/                 server FastAPI + pagina (index.html, app.js, stile.css, esporta.js, stampa.html)
scripts/               inizializzazione, verifica dell'ambiente, valutazioni ed esperimenti
tests/                 test automatici (pytest) e domande di prova del tutor
requirements.txt       dipendenze con versioni fissate
```

---

## Licenze e crediti

Componenti principali (licenze dichiarate dai pacchetti installati):

| Componente | Uso | Licenza |
|---|---|---|
| [datapizza-ai](https://pypi.org/project/datapizza-ai/) 0.1.0 | framework (LLM, RAG, agenti, tracing, MCP) | MIT |
| [Demucs](https://github.com/adefossez/demucs) 4.1.0, modello `htdemucs_6s` | separazione della chitarra | codice MIT; licenza dei pesi non dichiarata |
| [Basic Pitch](https://github.com/spotify/basic-pitch) 0.4.0 (Spotify) | trascrizione delle note | Apache-2.0 |
| [beat_this](https://github.com/CPJKU/beat_this) 1.1.0 | battiti e battute | MIT |
| [librosa](https://librosa.org) | tonalità, analisi audio | ISC |
| [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) 0.11 | scrittura dei file `.gp5` | LGPL-3.0 (usato come libreria, non modificato) |
| [alphaTab](https://alphatab.net) 1.8.4 | disegno e ascolto della tablatura (da CDN) | MPL-2.0 |
| [GeneralUser GS](http://www.schristiancollins.com) 1.471 (S. Christian Collins) | suoni dell'anteprima (copia locale) | licenza propria v2.0 (uso libero anche nel software; l'autore segnala di non poter garantire l'origine di tutti i campioni) — testo in `app/web/static/suoni/GeneralUser-LICENSE.txt` |
| [Qdrant client](https://github.com/qdrant/qdrant-client) | vector store del tutor (locale) | Apache-2.0 |
| [FastAPI](https://fastapi.tiangolo.com) / Uvicorn | pagina web | MIT / BSD-3-Clause |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) 1.30.0 | server MCP | MIT |
| [PyTorch](https://pytorch.org), onnxruntime, numpy, scipy, scikit-learn | calcolo | licenze permissive (vedi i pacchetti) |

Facoltativi / solo esperimenti:
- **BS-RoFormer** (`bs-roformer-infer` 0.1.5, MIT): pesi del modello SW di licenza **non dichiarata**.
- **PANNs** (`panns-inference` 0.1.1, MIT; pesi `Cnn14_DecisionLevelMax` da Zenodo, **CC BY 4.0** — Kong et al.,
  *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*, 2020): usato solo negli
  script di esperimento (Ispettore).

Modelli OpenAI (`gpt-5.4-nano`, `text-embedding-3-small`) usati via API con la chiave dell'utente.

Le **registrazioni MP3** dei brani di prova, gli **screenshot delle tablature di riferimento** e le **note trascritte**
da quelle tablature (`scripts/riferimento_*.py`, `tests/audio/reale/gt_*.json`) non sono inclusi nel repository
(esclusi da git): sono solo per uso personale e di studio. Senza questi file i test che li usano vengono saltati e gli
script di valutazione sui brani reali (`scripts/valuta_*`, `scripts/esperimento_*` che li importano) non partono; i
risultati delle misure sono riportati sopra.
