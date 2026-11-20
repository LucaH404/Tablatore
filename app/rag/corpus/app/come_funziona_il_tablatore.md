# Come funziona il Tablatore AI

Il Tablatore trasforma un MP3 in una tablatura per chitarra con una sequenza fissa di passi. I passi numerici sono **codice deterministico** (modelli locali, nessun LLM); l'LLM scrive solo il resoconto finale e risponde come tutor.

## I passi
1. **Validazione**: controlla che il file sia un MP3 valido, non troppo grande (massimo 20 MB) né troppo lungo (massimo 6 minuti), e non muto.
2. **Separazione**: isola la chitarra dal resto del brano con Demucs (modello `htdemucs_6s`, sei strumenti) o, in via sperimentale, con BS-RoFormer. Salva anche il basso e la chitarra isolata in MP3 (scaricabile, per ascoltare cosa "sente" l'app).
3. **Ritmo**: rileva battiti, battute, tempo (BPM) e tonalità sul brano **originale** (con beat_this e librosa). Senza batteria e sopra 130 BPM il tempo viene dimezzato.
4. **Trascrizione**: trasforma la chitarra isolata in note (Basic Pitch di Spotify). Parte con sensibilità media e, se trova meno di una nota per battito, ripete con sensibilità alta.
   - 4b **Power chord**: sulla chitarra distorta ricostruisce gli accordi interi con il basso (o dalla chitarra stessa); se i power chord sono spenti (chitarra pulita) corregge le note lette un'ottava sopra.
   - 4c **Ripetizioni** (opzione): confronta le battute che si ripetono e corregge per consenso.
   - 4d **Semplifica** (opzione): unisce gli accordi riattaccati mentre suonano ancora.
5. **Tablatura**: mette le note su una griglia di semicrome, sceglie l'accordatura e assegna corda e tasto a ogni nota con una programmazione dinamica che minimizza gli spostamenti e gli stiramenti della mano.
6. **Verifica**: il Verificatore controlla che la tab sia **suonabile** (estensione della mano, note sulla stessa corda, tasti esistenti) e toglie le note probabilmente false (armoniche spurie).
7. **Resoconto**: un LLM scrive 3-4 frasi per il chitarrista usando **solo** i numeri del risultato; un controllo automatico scarta il resoconto se contiene numeri che non compaiono nei dati.

## Osservabilità
Ogni passo è uno span di tracing (ContextTracing di datapizza-ai) con durata, decisioni, avvisi ed eventuali errori, salvato in `traccia.json` (scaricabile come "Traccia").
