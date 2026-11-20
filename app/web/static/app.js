// Logica della pagina: caricamento, avanzamento, risultato, lettore e esportazioni.
const $ = id => document.getElementById(id);
let opzioni = null, idLavoro = null, risultato = null, api = null, datiGp5 = null, inizio = null;
const AVVISI_FISSI_INIZIO = ["La trascrizione è generata automaticamente", "Le tablature di brani protetti"];

// ---------------------------------------------------------------- tema chiaro/scuro
// Senza scelta segue il sistema; il pulsante fissa l'opposto di quello che si vede e lo ricorda.
const scuroDiSistema = window.matchMedia("(prefers-color-scheme: dark)");
function temaAttuale() { return document.documentElement.dataset.theme || (scuroDiSistema.matches ? "dark" : "light"); }
function aggiornaPulsanteTema() {
  const scuro = temaAttuale() === "dark";
  $("tema").textContent = scuro ? "☀ Chiaro" : "☾ Scuro";
  $("tema").setAttribute("aria-label", scuro ? "Passa al tema chiaro" : "Passa al tema scuro");
}
$("tema").onclick = () => {
  const nuovo = temaAttuale() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nuovo;
  try { localStorage.setItem("tema", nuovo); } catch (e) { /* memoria del browser non disponibile */ }
  aggiornaPulsanteTema();
};
scuroDiSistema.addEventListener?.("change", aggiornaPulsanteTema);
aggiornaPulsanteTema();

async function json(url, init) {
  const r = await fetch(url, init);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `Errore ${r.status}`);
  return d;
}

function riempiSelect(sel, valori, scelto, etichetta = v => v) {
  sel.innerHTML = valori.map(v => `<option value="${v}" ${v === scelto ? "selected" : ""}>${etichetta(v)}</option>`).join("");
}

const NOMI_SEPARATORE = { demucs: "Demucs (predefinito)", roformer: "RoFormer (sperimentale)" };
const NOMI_STILE = { auto: "Automatico", power_chord: "Riff con power chord (rock, metal)", libero: "Note libere (arpeggi, accordi)" };
function nomeStile(v) { return NOMI_STILE[v] || v; }

function nomeAccordatura(v) {
  if (v === "auto") return "Automatica";
  const nomi = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const corde = (opzioni.corde[v] || []).map(m => nomi[m % 12]).join(" ");
  return `${v} (${corde})`;
}

// ---------------------------------------------------------------- avvio
async function avvio() {
  opzioni = await json("/api/opzioni");
  riempiSelect($("accordatura"), opzioni.accordature, "auto", nomeAccordatura);
  riempiSelect($("sensibilita"), opzioni.sensibilita, "auto", v => v === "auto" ? "Automatica" : v);
  riempiSelect($("stile"), opzioni.stili, "auto", nomeStile);
  // un server avviato prima dell'aggiornamento non manda "separatori": la pagina resta usabile
  riempiSelect($("separatore"), opzioni.separatori || ["demucs"], "demucs", v => NOMI_SEPARATORE[v] || v);
  $("limite-file").textContent = `Solo .mp3, massimo ${opzioni.dimensione_max_mb} MB e 6 minuti`;
  elencoPassi(opzioni.passi);
  if (opzioni.tutor === false) {                          // senza chiave OpenAI il tutor non può rispondere
    $("tutor-intro").textContent = "Tutor non disponibile: manca la chiave OPENAI_API_KEY nel file .env.";
    $("tutor-modulo").hidden = $("tutor-esempi").hidden = true;
  }
  const daUrl = new URLSearchParams(location.search).get("lavoro");
  if (daUrl) { idLavoro = daUrl; mostra("avanzamento"); segui(); }
}

function elencoPassi(passi) {
  $("passi").innerHTML = passi.map(p => `<li data-passo="${p}">${p.replace(/^\d+[a-z]? /, "")}</li>`).join("");
}

function mostra(sezione) {
  for (const s of ["carica", "avanzamento", "stop", "risultato"]) $(`sezione-${s}`).hidden = s !== sezione;
}

// ---------------------------------------------------------------- caricamento
const zona = $("zona-file");
$("file").addEventListener("change", () => { $("nome-file").textContent = $("file").files[0]?.name || ""; });
["dragover", "dragenter"].forEach(e => zona.addEventListener(e, ev => { ev.preventDefault(); zona.classList.add("sopra"); }));
["dragleave", "drop"].forEach(e => zona.addEventListener(e, () => zona.classList.remove("sopra")));
zona.addEventListener("drop", ev => {
  ev.preventDefault();
  if (ev.dataTransfer.files.length) { $("file").files = ev.dataTransfer.files; $("file").dispatchEvent(new Event("change")); }
});

$("modulo").addEventListener("submit", async ev => {
  ev.preventDefault();
  const f = $("file").files[0];
  $("errore-carica").hidden = true;
  if (!f) return;
  if (!f.name.toLowerCase().endsWith(".mp3")) return errore("Sono ammessi solo file .mp3.");
  if (f.size > opzioni.dimensione_max_mb * 1048576) return errore(`File troppo grande (massimo ${opzioni.dimensione_max_mb} MB).`);
  const dati = new FormData();
  dati.append("file", f);
  dati.append("accordatura", $("accordatura").value);
  dati.append("sensibilita", $("sensibilita").value);
  dati.append("stile", $("stile").value);
  dati.append("resoconto_llm", $("resoconto").checked);
  dati.append("separatore", $("separatore").value);
  dati.append("ripetizioni", $("ripetizioni").checked);
  dati.append("semplifica", $("semplifica").checked);
  elencoPassi(opzioni.passi.filter(p => (p !== "4c ripetizioni" || $("ripetizioni").checked)
                                     && (p !== "4d semplifica" || $("semplifica").checked)));
  $("avvia").disabled = true;
  try {
    idLavoro = (await json("/api/lavori", { method: "POST", body: dati })).id;
    history.replaceState(null, "", `/?lavoro=${idLavoro}`);       // ricaricando la pagina si ritrova il lavoro
    mostra("avanzamento");
    segui();
  } catch (e) { errore(e.message); }
  $("avvia").disabled = false;
});

function errore(msg) { $("errore-carica").textContent = msg; $("errore-carica").hidden = false; }

// ---------------------------------------------------------------- avanzamento
async function segui() {
  inizio = inizio || Date.now();
  let stato;
  try { stato = await json(`/api/lavori/${idLavoro}`); }
  catch (e) { mostra("carica"); return errore(e.message); }
  $("cronometro").textContent = `${Math.round((Date.now() - inizio) / 1000)} s`;
  $("coda").hidden = stato.stato !== "in_coda";
  $("coda").textContent = `In coda: ${stato.davanti_in_coda || 0} brani prima di questo.`;
  const passi = [...document.querySelectorAll("#passi li")];
  const i = passi.findIndex(li => li.dataset.passo === stato.passo);
  passi.forEach((li, k) => { li.className = i < 0 ? "" : k < i ? "fatto" : k === i ? "in-corso" : ""; });
  if (stato.stato === "finito") return mostraRisultato(stato);
  if (stato.stato === "errore") { mostra("stop"); $("motivo-stop").textContent = stato.errore; return; }
  setTimeout(segui, 700);
}

// ---------------------------------------------------------------- risultato
function mostraRisultato(stato) {
  risultato = stato.risultato;
  if (!risultato.completato) {
    mostra("stop");
    $("motivo-stop").textContent = risultato.motivo_stop;
    $("avvisi-stop").innerHTML = elencoAvvisi(risultato.avvisi);
    return;
  }
  mostra("risultato");
  document.title = `${stato.titolo} – Tablatore AI`;
  $("titolo-brano").textContent = stato.titolo;
  const fattore = risultato.fattore_tempo || 1;
  const rilevato = risultato.bpm_rilevato || risultato.bpm / fattore;
  const tempo = `${Math.round(risultato.bpm)} BPM` + (fattore !== 1 ? ` (rilevato ${Math.round(rilevato)})` : "");
  const pc = risultato.power_chord || {};
  const sempl = risultato.semplifica || {};
  const dati = [["Accordatura", risultato.accordatura], ["Tempo", tempo],
    ["Metro", risultato.metro], ["Tonalità", (risultato.tonalita || "—").replace(" (relativa", " · rel.").replace(")", "")],
    ["Battute", risultato.numero_battute], ["Note", risultato.numero_note], ["Sensibilità", risultato.sensibilita],
    ["Suonabile", risultato.suonabile ? "sì" : "no"],
    ["Power chord", pc.attiva ? (pc.fonte === "chitarra" ? "ricostruiti dalla chitarra" : "ricostruiti col basso") : "no"],
    ["Semplifica", sempl.attiva ? `${sempl.accordi_uniti} accordi uniti` : "no"],
    ["Separatore", NOMI_SEPARATORE[risultato.separatore || "demucs"].replace(/ \(.*\)/, "")],
    ["Ripetizioni", risultato.ripetizioni && risultato.ripetizioni.attiva
      ? `+${risultato.ripetizioni.note_aggiunte} / −${risultato.ripetizioni.note_tolte} note` : "no"]];
  $("dati").innerHTML = dati.map(([k, v]) => `<span class="etichetta">${k} <b>${esc(v)}</b></span>`).join("");
  $("avvisi").innerHTML = elencoAvvisi(risultato.avvisi);
  $("resoconto-testo").textContent = risultato.resoconto || "";
  $("decisioni").innerHTML = (risultato.decisioni.length ? risultato.decisioni : ["Nessuna: impostazioni di partenza."])
    .map(d => `<li>${esc(d)}</li>`).join("");
  $("consumi").textContent = `Tempo totale ${risultato.tempo_totale_s} s · token LLM ${risultato.token_in} + ${risultato.token_out} · costo $${risultato.costo_usd.toFixed(4)}`;
  riempiSelect($("nuova-accordatura"), opzioni.accordature.filter(a => a !== "auto"), risultato.accordatura, nomeAccordatura);
  riempiSelect($("nuovo-stile"), opzioni.stili, risultato.stile || "auto", nomeStile);
  $("nuove-ripetizioni").checked = !!(risultato.ripetizioni && risultato.ripetizioni.attiva);
  $("nuova-semplifica").checked = !!sempl.attiva;
  const NOMI_TEMPO = { 0.5: "metà", 1: "rilevato", 2: "doppio" };
  riempiSelect($("nuovo-tempo"), (opzioni.fattori_tempo || [1]).map(String), String(fattore),
               f => `${NOMI_TEMPO[f] || f} (${Math.round(rilevato * parseFloat(f))} BPM)`);
  const nomi = { "tablatura_verificata.gp5": "Guitar Pro (.gp5)", "note.mid": "MIDI", "tablatura_verificata.txt": "Testo",
                 "traccia.json": "Traccia", "chitarra.mp3": "Chitarra isolata (.mp3)" };
  $("scaricamenti").innerHTML = Object.entries(risultato.file).filter(([n]) => nomi[n])
    .map(([n, url]) => `<a href="${url}" download>${nomi[n]}</a>`).join("");
  caricaTablatura();
}

function elencoAvvisi(avvisi) {
  return avvisi.map(a => `<li class="${AVVISI_FISSI_INIZIO.some(f => a.startsWith(f)) ? "fisso" : ""}">${esc(a)}</li>`).join("");
}

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------------------------------------------------------------- alphaTab
async function caricaTablatura() {
  const url = risultato.file["tablatura_verificata.gp5"] + `?v=${Date.now()}`;
  datiGp5 = new Uint8Array(await (await fetch(url)).arrayBuffer());
  $("caricamento-tab").hidden = false;
  if (api) { api.load(datiGp5); return; }
  api = new alphaTab.AlphaTabApi($("alphatab"), {
    // senza caricamento "pigro": si disegna tutto subito (anche fuori schermo), così stampa,
    // screenshot e scorrimento durante l'ascolto trovano sempre la tablatura completa
    // e nel thread principale (useWorkers: false): con il worker, nelle prove con Chrome headless
    // la tablatura non veniva disegnata; i brani sono corti (max 6 min), il costo è trascurabile
    core: { fontDirectory: CDN_ALPHATAB + "font/", enableLazyLoading: false, useWorkers: false },
    display: { staveProfile: $("vista").value, layoutMode: "Page" },
    player: { playerMode: "EnabledSynthesizer", enableCursor: true, enableUserInteraction: true,
              // banco locale più realistico (GeneralUser GS, 31 MB) se installato, altrimenti quello piccolo del CDN
              soundFont: opzioni?.banco_suoni || CDN_ALPHATAB + "soundfont/sonivox.sf2",
              scrollElement: document.scrollingElement },
  });
  if (opzioni?.banco_suoni)                               // il banco grande impiega qualche secondo la prima volta
    $("stato-player").textContent = "suoni di chitarra in caricamento (31 MB, solo la prima volta)…";
  api.renderFinished.on(() => { $("caricamento-tab").hidden = true; margineScorrimento(); });
  api.error.on(e => { $("caricamento-tab").hidden = false; $("caricamento-tab").textContent = "Errore nel disegno: " + (e?.message || e); });
  api.playerReady.on(() => { $("play").disabled = $("stop").disabled = false; $("stato-player").textContent = ""; });
  api.soundFontLoadFailed?.on(() => { $("stato-player").textContent = "suoni non disponibili (serve la connessione)"; });
  api.playerStateChanged.on(e => { $("play").textContent = e.state === 1 ? "❚❚ Pausa" : "▶ Ascolta"; });
  api.scoreLoaded.on(score => impostaSuono(score));       // ogni nuova tablatura usa il suono scelto
  api.load(datiGp5);
}

// ---------------------------------------------------------------- suono dell'anteprima
// Lo strumento è il "program" General MIDI della traccia (25 = chitarra acustica in acciaio,
// 30 = chitarra distorta): si cambia nel browser e si rigenera solo il MIDI dell'anteprima.
const CHIAVE_SUONO = "tab-suono";
try { const s = localStorage.getItem(CHIAVE_SUONO); if (s && $("suono").querySelector(`option[value="${s}"]`)) $("suono").value = s; }
catch (e) { /* memoria del browser non disponibile: si usa il suono predefinito */ }

function impostaSuono(score) {
  const program = parseInt($("suono").value, 10);
  const STRUMENTO = alphaTab.model.AutomationType.Instrument;
  for (const t of score?.tracks || []) {
    t.playbackInfo.program = program;
    // al caricamento alphaTab copia lo strumento anche in un cambio di strumento sulla prima nota,
    // che nel MIDI vince sul program della traccia: vanno cambiati anche quelli
    for (const st of t.staves) for (const b of st.bars) for (const v of b.voices) for (const beat of v.beats)
      for (const a of beat.automations) if (a.type === STRUMENTO) a.value = program;
  }
}

$("suono").onchange = () => {
  try { localStorage.setItem(CHIAVE_SUONO, $("suono").value); } catch (e) { /* ignora */ }
  if (!api || !api.score) return;
  impostaSuono(api.score);
  api.loadMidiForScore();                                 // ferma la riproduzione e rigenera il MIDI
  // niente ripartenza automatica: in alphaTab 1.8.4 anche solo leggere api.midiLoaded manda la
  // pagina in "Maximum call stack size exceeded" (provato in Chrome headless)
  $("stato-player").textContent = "suono cambiato: premi Ascolta";
};

// Scorrimento automatico: alphaTab porta la riga suonata in cima alla finestra, ma lì c'è la barra
// degli strumenti fissa (position: sticky) che la copre. Spostamento negativo = la riga si ferma
// sotto la barra. L'altezza della barra cambia con la larghezza (i controlli vanno a capo): si
// ricalcola prima di ogni ascolto e quando la finestra cambia dimensione.
function margineScorrimento() {
  if (!api) return;
  const barra = document.querySelector(".barra-strumenti");
  const fissa = barra && getComputedStyle(barra).position === "sticky";
  // letto da alphaTab a ogni scorrimento (doScroll): non serve updateSettings()
  api.settings.player.scrollOffsetY = -((fissa ? barra.offsetHeight : 0) + 16);
}
window.addEventListener("resize", () => margineScorrimento());

$("play").onclick = () => { if (!api) return; margineScorrimento(); api.playPause(); };
$("stop").onclick = () => api && api.stop();
$("velocita").onchange = () => { if (api) api.playbackSpeed = parseFloat($("velocita").value); };
$("vista").onchange = () => {
  if (!api) return;
  api.settings.display.staveProfile = alphaTab.StaveProfile[$("vista").value];
  api.updateSettings();
  api.render();
};

$("riaccorda").onclick = async () => {
  const dati = new FormData();
  dati.append("accordatura", $("nuova-accordatura").value);
  dati.append("stile", $("nuovo-stile").value);
  dati.append("ripetizioni", $("nuove-ripetizioni").checked);
  dati.append("semplifica", $("nuova-semplifica").checked);
  dati.append("tempo", $("nuovo-tempo").value);
  const pulsante = $("riaccorda"), testo = pulsante.textContent;
  pulsante.disabled = true;
  pulsante.classList.add("in-attesa");                    // rotellina finché il server risponde
  pulsante.textContent = "Rifaccio la tablatura…";
  try { mostraRisultato(await json(`/api/lavori/${idLavoro}/accordatura`, { method: "POST", body: dati })); }
  catch (e) { alert(e.message); }
  finally {
    pulsante.classList.remove("in-attesa");
    pulsante.textContent = testo;
    pulsante.disabled = false;
  }
};

$("pdf").onclick = () => window.open(`/stampa?lavoro=${idLavoro}&vista=${$("vista").value}`, "_blank");
$("png").onclick = async () => {
  $("png").disabled = true;
  try { await esportaPng(datiGp5, `${$("titolo-brano").textContent} - tablatura.png`, $("vista").value); }
  catch (e) { alert("Esportazione non riuscita: " + (e?.message || e)); }
  $("png").disabled = false;
};

// ---------------------------------------------------------------- tutor di teoria (RAG)
// La conversazione resta nel browser (storia): il server non conserva nulla e riceve ogni volta
// gli ultimi messaggi, più l'id del lavoro aperto per dare al tutor i dati della tablatura.
let storiaTutor = [];
const RIGA_TAB = /^\s*[eBGDAE]\|/;

function messaggioTutor(ruolo, testo, extra = {}) {
  const div = document.createElement("div");
  div.className = `msg ${ruolo}` + (extra.classe ? ` ${extra.classe}` : "");
  if (ruolo === "tutor" && !extra.classe) {
    // righe di tablatura in carattere a larghezza fissa; il resto come testo semplice
    div.innerHTML = String(testo).split("\n").map(r => RIGA_TAB.test(r)
      ? `<span class="tab-riga">${esc(r)}</span>` : esc(r)).join("\n");
    if (extra.fonti?.length) {
      const f = document.createElement("span");
      f.className = "fonti";
      f.textContent = "Schede consultate: " + extra.fonti.join(", ") + (extra.costo ? ` · $${extra.costo.toFixed(5)}` : "");
      div.appendChild(f);
    }
  } else {
    div.textContent = testo;
  }
  $("tutor-messaggi").appendChild(div);
  div.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return div;
}

async function chiediTutor(domanda) {
  domanda = domanda.trim();
  if (!domanda) return;
  $("tutor-domanda").value = "";
  $("tutor-esempi").hidden = true;
  $("tutor-nuova").hidden = false;
  messaggioTutor("utente", domanda);
  const attesa = messaggioTutor("tutor", "Il tutor sta consultando le schede…", { classe: "in-attesa" });
  $("tutor-invia").disabled = true;
  try {
    const r = await json("/api/tutor", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domanda, storia: storiaTutor, id_lavoro: risultato ? idLavoro : null }),
    });
    attesa.remove();
    const rifiuto = Boolean(r.rifiutata);
    messaggioTutor("tutor", r.risposta, rifiuto ? { classe: "rifiuto" } : { fonti: r.citate?.length ? r.citate : r.fonti, costo: r.costo_usd });
    if (!rifiuto) storiaTutor.push({ ruolo: "utente", testo: domanda }, { ruolo: "tutor", testo: r.risposta });
    storiaTutor = storiaTutor.slice(-12);
    $("tutor-stato").textContent = r.scartata ? "Risposta originale scartata dal controllo delle fonti: mostrato un testo di riserva." : "";
  } catch (e) {
    attesa.remove();
    messaggioTutor("tutor", e.message, { classe: "rifiuto" });
  } finally {
    $("tutor-invia").disabled = false;
    $("tutor-domanda").focus();
  }
}

$("tutor-modulo").onsubmit = ev => { ev.preventDefault(); chiediTutor($("tutor-domanda").value); };
for (const b of document.querySelectorAll("#tutor-esempi .esempio")) b.onclick = () => chiediTutor(b.textContent);
$("tutor-nuova").onclick = () => {
  storiaTutor = [];
  $("tutor-messaggi").replaceChildren();
  $("tutor-esempi").hidden = false;
  $("tutor-nuova").hidden = true;
  $("tutor-stato").textContent = "";
};

avvio();
