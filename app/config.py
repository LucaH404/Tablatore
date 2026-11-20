"""Parametri del progetto in un unico punto, così sono facili da trovare e cambiare."""

# --- Guardiano dell'input: limiti sul file caricato ---
ESTENSIONI_AMMESSE = {".mp3"}
DIMENSIONE_MAX_MB = 20          # ~20 minuti di MP3 a 128 kbps: ampio margine sopra la durata massima
DURATA_MIN_S = 3.0              # sotto i 3 s non c'è abbastanza musica per stimare tempo e tonalità
DURATA_MAX_S = 6 * 60.0         # oltre 6 min separazione e trascrizione diventano lente e meno affidabili
BITRATE_MIN_CONSIGLIATO = 96_000   # sotto questa soglia l'MP3 perde dettagli utili alla trascrizione
SAMPLE_RATE_MIN_CONSIGLIATO = 22_050
# Silenzi: le registrazioni manuali (es. NVIDIA Instant Replay) hanno momenti morti. "Nine thou":
# 3,7-3,9 s iniziali a circa -80 dB; l'intro più debole del brano reale è a circa -16 dB.
SOGLIA_SILENZIO_DB = -50         # assoluti (dBFS): ampio margine sopra il silenzio digitale
SILENZIO_MIN_S = 2.0             # pause più brevi fanno parte della musica

# --- Analista audio: separazione della chitarra (Demucs) ---
MODELLO_DEMUCS = "htdemucs_6s"   # unico modello Demucs con una traccia dedicata alla chitarra
SAMPLE_RATE_LAVORO = 44_100      # frequenza nativa di Demucs: si decodifica direttamente a questa
DEMUCS_SHIFTS = 0                # 0 = risultato deterministico (con 1, default di Demucs, la quota
                                 # di chitarra oscillava tra 62% e 93% sullo stesso file); stesso tempo
QUOTA_CHITARRA_MIN = 0.03        # sotto il 3% dell'energia totale la chitarra è considerata assente
TRACCE_ACCOMPAGNAMENTO = ("drums", "bass", "vocals", "piano")
QUOTA_ACCOMPAGNAMENTO_MAX_SOLO = 0.15  # accompagnamento < 15% -> il file è già solo chitarra
                                       # (misurato: solo chitarra 6%, qualsiasi mix 75-100%)
CARTELLA_OUTPUT = "output"
# Separatore alternativo (SPERIMENTALE, Step 17): BS-RoFormer SW, 6 tracce come htdemucs_6s.
# Su Nine Thou intero: tablatura come Songsterr 64 -> 70%. Ma sul brano sintetico la chitarra
# finisce nel basso (stem chitarra a -90 dB): se la sua quota è sotto QUOTA_CHITARRA_MIN si
# ripiega su Demucs. Pesi da 700 MB a licenza sconosciuta (O61): per questo non è il default.
SEPARATORI = ("demucs", "roformer")
MODELLO_ROFORMER = "roformer-model-bs-roformer-sw-by-jarredou"

# --- Analista audio: trascrizione delle note (Basic Pitch, modello ONNX) ---
# Estensione della chitarra: dal Si1 (61,7 Hz, MIDI 35, corda grave in drop B) al 24° tasto
# del Mi cantino (Mi6, 1318 Hz, MIDI 88). Le frequenze fuori range sono scartate.
# (Prima il minimo era Mi2 = 80 Hz, l'accordatura standard: "Nine thou" è in drop C#, con la
# corda grave a Do#2 = 69 Hz, che sarebbe stata scartata. Sui test sintetici nessun effetto.)
FREQ_MIN_HZ = 60.0
FREQ_MAX_HZ = 1400.0
MIDI_MIN_CHITARRA = 35
MIDI_MAX_CHITARRA = 88
# Soglie (attacco, durata, nota minima). Nessuna soglia va bene per tutto:
#  - chitarra pulita (brano sintetico di prova): con "media" F1 0,96 / 0,88; con soglie basse
#    0,49, perché le note che risuonano vengono ri-attaccate (doppioni);
#  - registrazione con effetti ("Nine thou", riff in semicrome): con "media" F1 0,29 (13 note
#    su 42), con "alta" 0,53 e 89% delle note acute trovate.
# Per questo la sensibilità si adatta: vedi NOTE_PER_BEAT_MIN.
SENSIBILITA = {
    "bassa": {"onset": 0.9, "frame": 0.5, "durata_min_ms": 128},  # meno note: se ne escono troppe
    "media": {"onset": 0.8, "frame": 0.5, "durata_min_ms": 128},  # default
    "alta":  {"onset": 0.5, "frame": 0.3, "durata_min_ms": 58},   # più note e più corte
}
# Sotto 1 nota per beat con "media" conviene riprovare con "alta". Misurato: brani di prova
# 2,05-2,20 note/beat (restano su "media"); "Nine thou" 0,83 (sintetizzato) e 0,42 (reale).
NOTE_PER_BEAT_MIN = 1.0
FPS_BASIC_PITCH = 22050 / 256    # frame al secondo dell'uscita del modello

# --- Analista audio: tempo, battute e tonalità (beat_this + librosa) ---
MODELLO_BEAT = "final0"          # checkpoint di default di beat_this (77 MB, cache di torch hub)
SAMPLE_RATE_BEAT = 22_050        # frequenza su cui beat_this è stato addestrato
# Il tempo va stimato sull'audio ORIGINALE: sulla sola chitarra (arpeggi in crome, senza
# batteria) beat_this ha dato 130 BPM invece di 100; sul mix 100,0 BPM esatti.
BPM_MIN_PLAUSIBILE = 40
BPM_MAX_PLAUSIBILE = 240
VARIAZIONE_TEMPO_MAX = 0.05      # oltre il 5% di variazione tra i beat: tempo non costante
QUOTA_INTERVALLI_ANOMALI_MAX = 0.20  # oltre il 20% di intervalli lontani (>15%) dal periodo, i beat
                                 # di beat_this sono incoerenti -> librosa come secondo parere.
                                 # (Prima: variazione su tutto il brano > 10%, che scattava anche
                                 # per 3 beat mancanti nell'intro del brano reale "Nine thou".)
# Metro dagli accenti gravi (cassa/basso sul primo tempo). Misurato: valzer 3/4 contrasto 2,43
# per gruppi da 3 contro 0,24; mix 4/4: 0,07 per 3 contro 0,67-0,73 per 2 e 4.
CONTRASTO_ACCENTI_MIN = 0.2      # sotto: nessun accento chiaro -> 4/4 di default, con avviso
RAPPORTO_TERZINARIO = 1.5        # 3/4 solo se il gruppo da 3 supera di 1,5 volte gli altri

# --- Power chord con il basso (Tool 4b) ---
# Misurato su tutto "Nine Thou": la trascrizione non trova mai le fondamentali gravi della chitarra
# distorta; il basso (Demucs) le suona nell'84% dei casi. Soglie di attivazione automatica stimate
# su 4 registrazioni (1 pulita, 3 distorte): sono PRUDENTI, perché sbagliare su un brano non fatto di
# power chord è disastroso (arpeggi: F1 da 0,88 a 0,17). Lo stile si può sempre forzare.
SR_BASSO = 11_025
HOP_BASSO = 128                  # 11,6 ms
MIDI_MIN_BASSO = 24              # C1
DINAMICA_BASSO_DB = 30           # il basso "suona" se è entro 30 dB dal suo massimo
SILENZIO_BASSO_DB = -80          # ...e sopra questa soglia assoluta (traccia muta = basso assente)
FINESTRA_ATTACCO_S = 0.05        # note attaccate insieme
PIATTEZZA_DISTORSIONE_MIN = 0.0003   # arpeggi puliti 0,0001; distorte 0,0005-0,0018
COERENZA_BASSO_MIN = 0.75        # Nine Thou 82-86%; arpeggi 77% ma puliti; clip sintetizzata 69%
QUOTA_BASSO_MIN = 0.15
ATTACCHI_CON_BASSO_MIN = 20
RADICE_POWER_CHORD_MIN = 36      # fondamentale tra C2 e B2 (registro delle corde gravi)
# ripiego senza basso (Step 18): fondamentale dedotta dalle note della chitarra stessa
COERENZA_INTERNA_MIN = 0.85      # accordi che stanno in un power chord: Lying From You 92%, Nine Thou 99-100%
ACCORDI_INTERNI_MIN = 20         # servono almeno 20 accordi (2+ note) per giudicare
OTTAVA_MIDI_MIN = 48             # Step 20 (O70): stessa nota a distanza di ottava, nota grave >= C3 = ottava, non power chord
                                 # (Lying From You tab 78 -> 90%; Nine Thou 63,8 -> 63,4 / 70,5 invariato; None = spenta)
FINESTRA_OTTAVA_S = 0.4         # nota singola entro 0,4 s da un'ottava della stessa nota = ottava
BONUS_FORMA_OTTAVA = 1.5         # diteggiatura: ottava con nota grave su 5a/6a corda e una corda saltata
# Step 22 (O73): note lette un'ottava sopra su chitarra pulita (solo con power chord spenti)
OTTAVA_SU_FINESTRA_S = 3.0       # finestra in cui cercare la stessa nota un'ottava sotto
OTTAVA_SU_MIN_SOTTO = 2          # quante volte deve comparire l'ottava sotto
OTTAVA_SU_PERCENTILE = 80        # la nota deve essere tra le più acute della finestra
FINESTRA_QUINTA_S = 0.35         # quinta isolata entro 0,35 s dall'accordo = stesso accordo

# --- Consenso tra ripetizioni (Tool 4c, SPERIMENTALE, Step 17) ---
# Battute simili (cromagramma a semicrome di chitarra e basso) votano le note. Su Nine Thou:
# +3-4 punti di richiamo, tablatura 64 -> 66% (Demucs) e 70 -> 75% (RoFormer); tetto del
# metodo con gruppi perfetti 77%, perché gli errori della trascrizione sono soprattutto sistematici.
SIMILARITA_RIPETIZIONI = 0.8     # somiglianza minima tra due battute (coseno)
GRUPPO_RIPETIZIONI_MIN = 3       # la battuta + almeno 2 ripetizioni
VICINI_RIPETIZIONI_MAX = 12
# Step 19: tempo e note extra
FATTORI_TEMPO = (0.5, 1.0, 2.0)  # tempo della tablatura rispetto a quello rilevato
BPM_DIMEZZA_SENZA_BATTERIA = 130 # senza batteria e sopra 130 BPM il tempo si dimezza (Lying From You: 159 -> 80)
SOGLIA_UNISCI_BEAT = 0.5         # "semplifica": stesso accordo riattaccato entro una croma mentre suona -> uno solo
BASSO_UDIBILE_DB = -30           # basso usato nel confronto solo se entro 30 dB dalla chitarra (con basso: -7..+2)
QUOTA_AGGIUNGI = 0.5             # nota presente in metà del gruppo -> aggiunta dove manca
QUOTA_TOGLI = 0.25               # nota presente in meno di 1/4 del gruppo -> tolta
TOLLERANZA_POSIZIONE_BEAT = 0.15 # stessa posizione nella battuta (frazione di beat)

# --- Tablatore: dalle note alla tablatura ---
# Accordature (MIDI delle corde a vuoto, dalla 6a grave alla 1a acuta) e "costo" di preferenza:
# a parità di diteggiatura vince la più comune. "Nine thou" è in drop C#.
ACCORDATURE = {
    "standard":     ([40, 45, 50, 55, 59, 64], 0.0),
    "drop D":       ([38, 45, 50, 55, 59, 64], 1.0),
    "Eb standard":  ([39, 44, 49, 54, 58, 63], 1.0),
    "D standard":   ([38, 43, 48, 53, 57, 62], 2.0),
    "drop C#":      ([37, 44, 49, 54, 58, 63], 2.0),
    "drop C":       ([36, 43, 48, 53, 57, 62], 2.0),
    "C# standard":  ([37, 42, 47, 52, 56, 61], 3.0),
    "C standard":   ([36, 41, 46, 51, 55, 60], 3.0),
    "drop B":       ([35, 42, 47, 52, 56, 61], 3.0),
}
TASTI = 24
SUDDIVISIONI_BEAT = 4            # griglia a semicrome (terzine non gestite)
ESTENSIONE_MANO_MAX = 5          # tasti tra il più basso e il più alto in un accordo
# Scelta automatica dell'accordatura: si tollerano poche note fuori estensione (probabili note
# false della trascrizione) prima di passare a un'accordatura meno comune.
NOTE_FUORI_TOLLERATE = 2
QUOTA_FUORI_TOLLERATA = 0.02
PAUSA_MIN_SLOT = 4               # buchi più corti di 1 beat: la nota viene allungata (più leggibile)
BONUS_FORMA_A_UN_DITO = 0.3      # power chord a un dito delle accordature drop (vedi _diteggiature)

# --- Verificatore di suonabilità ---
# Note isolate sospette (armoniche/effetti scambiati per note). Misurato nelle finestre con
# riferimento Songsterr: con ">= 10 semitoni sopra la mediana delle vicine E nessuna vicina nel
# suo registro" -> "Nine thou" reale 7 note, tutte false; sintetizzato 1, falsa; note esatte
# (oracolo) 0. Allargando a "al massimo il 20% delle vicine nel registro" sul reale se ne
# trovano 9 (false), ma sull'oracolo si toglierebbero 5 note VERE: il riff alterna power chord
# gravi e note un'ottava sopra. Il criterio resta prudente.
FINESTRA_CONTESTO_BEAT = 1.0     # note vicine: entro 1 beat prima e dopo
SALTO_REGISTRO_SOSPETTO = 10     # semitoni sopra la mediana delle vicine
REGISTRO_VICINO = 5              # ...e nessuna vicina entro 5 semitoni (un assolo acuto non è sospetto)
SALTO_MANO_MIN_TASTI = 5         # spostamenti più piccoli non si segnalano
# Velocità della mano: stima fisica, non tarata su dati (le tablature di riferimento non hanno
# salti >= 5 tasti). 60 tasti/s = 12 tasti richiedono almeno 200 ms, 6 tasti 100 ms.
VELOCITA_MANO_MAX = 60.0
CICLI_CORREZIONE_MAX = 3
QUOTA_NOTE_TOLTE_ALTA = 0.15     # oltre il 15% di note tolte: trascrizione poco affidabile

# --- Orchestratore ---
MODELLO_LLM = "gpt-5.4-nano"
PREZZI_LLM = {"gpt-5.4-nano": (0.20, 1.25)}   # $ per 1M token (in, out)
MAX_PASSI_AGENTE = 12            # modalità agente: 6 tool + un nuovo tentativo + risposta, con margine

# --- Tutor di teoria (RAG, Step 25) ---
TUTOR_CORPUS = "app/rag/corpus"          # schede Markdown (teoria / tecnica / app), testo originale
TUTOR_INDICE = "dati/indice_tutor"       # Qdrant locale su disco (git-ignorato; si ricrea con l'ingestione)
TUTOR_COLLEZIONE = "tutor_chitarra"
TUTOR_EMB_MODELLO = "text-embedding-3-small"   # 1536 dimensioni
TUTOR_EMB_DIM = 1536
TUTOR_EMB_NOME = "contenuto"
PREZZO_EMBEDDING = 0.02                  # $ per 1M token (text-embedding-3-small)
TUTOR_K = 4                              # passaggi recuperati per domanda
TUTOR_TURNI_MEMORIA = 6                  # ultimi messaggi (domande + risposte) ricordati
TUTOR_DOMANDA_MAX = 600                  # caratteri
TUTOR_SIMILITUDINE_MIN = 0.32            # sotto: domanda fuori tema. Valutazione Step 25: in tema min 0,345,
                                         # fuori tema max 0,299 (20 + 5 domande): margine stretto

# --- Pagina web (FastAPI + alphaTab) ---
HOST = "127.0.0.1"               # solo questo PC: la pagina non è esposta in rete
PORTA = 8000
CARTELLA_LAVORI = "output/lavori"   # una sottocartella per ogni brano caricato
BANCO_SUONI = "GeneralUser.sf2"  # anteprima: GeneralUser GS 1.471 (npm generaluser@1.47.1), in app/web/static/suoni/
ALPHATAB_VERSIONE = "1.8.4"      # MPL-2.0; versione esatta (verificata con i nostri .gp5, O44)

# --- Avvisi mostrati sempre all'utente ---
AVVISO_COPYRIGHT = (
    "Le tablature di brani protetti da copyright sono solo per uso personale e di studio: "
    "non vanno pubblicate né distribuite."
)
AVVISO_APPROSSIMAZIONE = (
    "La trascrizione è generata automaticamente ed è approssimativa: "
    "note, ritmo e diteggiature vanno verificati a orecchio."
)
