# Leggere avvisi e decisioni automatiche

## Avvisi
Gli avvisi arrivano direttamente dai passi della pipeline (non passano dall'LLM) e dicono **dove fidarsi meno**:
- **"chitarra poco presente"** (separazione): la chitarra pesa poco nel mix; le note saranno meno affidabili.
- **"Tempo dimezzato in automatico"** (ritmo): senza batteria il tempo rilevato era sopra 130 BPM ed è stato dimezzato. Se la tab sembra troppo lenta o con figure troppo lunghe, scegli "Tempo → rilevato" e rifai la tablatura.
- **Poche note per battito** (trascrizione): la trascrizione è stata ripetuta con sensibilità alta.
- **Note tolte dal Verificatore**: quante note erano probabilmente false (armoniche) o impossibili da suonare.
- **Semplificazione** e **ripetizioni**: quanti accordi o note sono stati uniti, aggiunti o tolti.
- Due avvisi fissi compaiono sempre: la trascrizione è automatica e approssimativa; le tab di brani protetti sono solo per uso personale e di studio.

## Decisioni automatiche
Nel riquadro "Decisioni automatiche e tempi" l'app scrive cosa ha scelto da sola e perché, per esempio:
- sensibilità ripetuta con "alta";
- separatore usato (RoFormer può ripiegare su Demucs);
- power chord ricostruiti con il basso o dalla chitarra, oppure non ricostruiti (e quante note un'ottava sopra sono state abbassate);
- accordatura scelta (o imposta dall'utente);
- tempo dimezzato o raddoppiato.

## Il resoconto
Le 3-4 frasi in alto sono scritte da un LLM a partire **solo dai dati** del risultato. Se l'LLM inventa un numero che non c'è nei dati, il resoconto viene scartato e sostituito da un testo scritto dal codice. Costo tipico: circa $0,0003 a brano.

## Cosa controllare per primo
1. Accordatura: se la chitarra del brano è accordata diversamente, tutta la tab è spostata.
2. Tempo: se le figure sembrano il doppio o la metà del dovuto.
3. Le sezioni segnalate negli avvisi.
