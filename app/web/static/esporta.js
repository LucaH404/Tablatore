// Esportazione della tablatura in immagini (PNG e pagine di stampa per il PDF).
// alphaTab disegna a pezzi posizionati in modo assoluto: per l'immagine si ricompongono, per la
// stampa ogni pezzo (un rigo) diventa un'immagine normale che il browser sa mandare a capo pagina
// senza tagliarla (verificato: la stampa diretta sovrapponeva i righi al cambio pagina).
const CDN_ALPHATAB = `https://cdn.jsdelivr.net/npm/@coderline/alphatab@${window.ALPHATAB_VERSIONE}/dist/`;

function disegnaSuCanvas(datiGp5, { larghezza = 800, nitidezza = 2, profilo = "Tab" } = {}) {
  return new Promise((risolvi, rifiuta) => {
    const div = document.createElement("div");
    div.style.cssText = `position:absolute;left:-20000px;top:0;width:${larghezza * nitidezza}px`;
    document.body.appendChild(div);
    const api = new alphaTab.AlphaTabApi(div, {
      core: { engine: "html5", useWorkers: false, enableLazyLoading: false, fontDirectory: CDN_ALPHATAB + "font/" },
      display: { staveProfile: profilo, layoutMode: "Page", scale: nitidezza },
    });
    api.error.on(e => { div.remove(); rifiuta(e); });
    api.renderFinished.on(() => setTimeout(() => {
      const base = div.getBoundingClientRect();
      const pezzi = [...div.querySelectorAll("canvas")].map(c => {
        const r = c.getBoundingClientRect();
        const copia = document.createElement("canvas");          // copia: l'originale sparisce con destroy()
        copia.width = c.width; copia.height = c.height;
        copia.getContext("2d").drawImage(c, 0, 0);
        return { canvas: copia, x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height };
      }).sort((a, b) => a.y - b.y || a.x - b.x);
      api.destroy();
      div.remove();
      risolvi(pezzi);
    }, 200));
    api.load(datiGp5);
  });
}

function unisciPezzi(pezzi) {
  const w = Math.max(...pezzi.map(p => p.x + p.w)), h = Math.max(...pezzi.map(p => p.y + p.h));
  const out = document.createElement("canvas");
  out.width = Math.ceil(w); out.height = Math.ceil(h);
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, out.width, out.height);
  pezzi.forEach(p => ctx.drawImage(p.canvas, p.x, p.y, p.w, p.h));
  return out;
}

async function esportaPng(datiGp5, nomeFile, profilo) {
  const tela = unisciPezzi(await disegnaSuCanvas(datiGp5, { larghezza: 1100, nitidezza: 2, profilo }));
  const a = document.createElement("a");
  a.href = tela.toDataURL("image/png");
  a.download = nomeFile;
  a.click();
}
