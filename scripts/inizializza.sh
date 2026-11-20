#!/usr/bin/env bash
# Inizializza il Tablatore AI su macOS o Linux: ambiente Python 3.11, dipendenze, ffmpeg, .env, modelli, verifica.
#
#   bash scripts/inizializza.sh [--cpu] [--sperimentali] [--senza-modelli] [--senza-test]
#
# Stessi passi di scripts/inizializza.ps1 (Windows). Si può rilanciare senza danni.
# Serve un python3 di sistema (3.8+) per installare uv; uv scarica da solo Python 3.11.
# Mac con processore Intel: NON supportati (Demucs lì richiede torch < 2.3 e numpy < 2).
# Nota: questo script è stato controllato solo nella sintassi; il progetto è stato provato su Windows.
set -u

CPU=0; SPERIMENTALI=0; SENZA_MODELLI=0; SENZA_TEST=0
for a in "$@"; do
  case "$a" in
    --cpu) CPU=1 ;; --sperimentali) SPERIMENTALI=1 ;; --senza-modelli) SENZA_MODELLI=1 ;; --senza-test) SENZA_TEST=1 ;;
    *) echo "Opzione sconosciuta: $a"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
VENV_PY="$ROOT/.venv/bin/python"
TORCH_VER="2.14.1"
INIZIO=$(date +%s)

passo() { printf '\n\033[36m[%s] %s\033[0m\n' "$1" "$2"; }
info() { echo "    $1"; }
fallisci() { printf '\n\033[31mERRORE: %s\033[0m\n' "$1"; exit 1; }
esegui() { local d="$1"; shift; "$@" || fallisci "$d"; }
chiedi_si() { local r; read -r -p "    $1 [S/n] " r; [[ -z "$r" || "$r" =~ ^[sSyY] ]]; }

echo "Tablatore AI - inizializzazione ($ROOT)"
SISTEMA="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$SISTEMA" == "Darwin" && "$ARCH" == "x86_64" ]]; then
  fallisci "Mac con processore Intel non supportato (Demucs richiede lì torch < 2.3 e numpy < 2)."
fi

# ---------------------------------------------------------------- 1. Python di sistema e uv
passo "1/9" "Python di sistema e uv"
SYS_PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    SYS_PY="$(command -v "$c")"; break
  fi
done
[[ -n "$SYS_PY" ]] || fallisci "nessun python3 (3.8+) trovato: installalo (es. 'brew install python' o 'sudo apt install python3') e rilancia."
info "Python di sistema: $SYS_PY"
if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif "$SYS_PY" -m uv --version >/dev/null 2>&1; then
  UV=("$SYS_PY" -m uv)
else
  info "installo uv per l'utente corrente..."
  if "$SYS_PY" -m pip install --user --upgrade uv; then
    UV=("$SYS_PY" -m uv)
  else
    fallisci "non riesco a installare uv con pip (Python 'gestito dal sistema'?). Installalo seguendo https://docs.astral.sh/uv/getting-started/installation/ e rilancia."
  fi
fi
info "uv: $("${UV[@]}" --version)"

# ---------------------------------------------------------------- 2. ambiente Python 3.11
passo "2/9" "Ambiente virtuale .venv con Python 3.11"
if [[ -x "$VENV_PY" ]]; then
  VER="$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [[ "$VER" == "3.11" ]] || fallisci ".venv esiste ma usa Python $VER: rinominala o cancellala e rilancia."
  info ".venv già presente (Python 3.11)"
else
  esegui "creazione del .venv" "${UV[@]}" venv --python 3.11 .venv
fi

# ---------------------------------------------------------------- 3. PyTorch
passo "3/9" "PyTorch $TORCH_VER"
if [[ "$SISTEMA" == "Darwin" ]]; then
  info "macOS (Apple Silicon): PyTorch da PyPI (accelerazione MPS non usata dall'app: si usa la CPU)"
  esegui "installazione di PyTorch" "${UV[@]}" pip install --python "$VENV_PY" "torch==$TORCH_VER"
else
  if [[ $CPU -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    INDICE=cu126; info "GPU NVIDIA trovata: versione CUDA 12.6 (download ~4-5 GB)"
  else
    INDICE=cpu; info "nessuna GPU NVIDIA (o --cpu): versione per CPU. La separazione sarà ~10 volte più lenta."
  fi
  esegui "installazione di PyTorch" "${UV[@]}" pip install --python "$VENV_PY" "torch==$TORCH_VER" \
    --index-url "https://download.pytorch.org/whl/$INDICE"
fi

# ---------------------------------------------------------------- 4. pacchetti senza dipendenze
passo "4/9" "Basic Pitch, beat_this, torchaudio (senza dipendenze)"
# basic-pitch con le dipendenze installerebbe TensorFlow; torchaudio/beat-this cambierebbero torch
esegui "pacchetti senza dipendenze" "${UV[@]}" pip install --python "$VENV_PY" --no-deps \
  "basic-pitch==0.4.0" "beat-this==1.1.0" "torchaudio==2.11.1" "rotary-embedding-torch==0.9.2"

# ---------------------------------------------------------------- 5. requirements.txt
passo "5/9" "Dipendenze del progetto (requirements.txt)"
esegui "requirements.txt" "${UV[@]}" pip install --python "$VENV_PY" -r requirements.txt
if [[ $SPERIMENTALI -eq 1 ]]; then
  info "separatore sperimentale BS-RoFormer..."
  esegui "bs-roformer-infer" "${UV[@]}" pip install --python "$VENV_PY" "bs-roformer-infer==0.1.5"
fi

# ---------------------------------------------------------------- 6. ffmpeg
passo "6/9" "ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
  info "ffmpeg già presente: $(command -v ffmpeg)"
elif [[ "$SISTEMA" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  if chiedi_si "ffmpeg manca. Lo installo con Homebrew?"; then esegui "installazione di ffmpeg" brew install ffmpeg; fi
elif command -v apt-get >/dev/null 2>&1; then
  if chiedi_si "ffmpeg manca. Lo installo con 'sudo apt-get install ffmpeg' (chiede la password)?"; then
    esegui "installazione di ffmpeg" sudo apt-get install -y ffmpeg
  fi
else
  info "ffmpeg manca: installalo con il gestore di pacchetti del sistema (https://ffmpeg.org/download.html)."
fi

# ---------------------------------------------------------------- 7. .env con la chiave OpenAI
passo "7/9" "File .env (chiave OpenAI)"
if [[ -f .env ]] && grep -qE '^OPENAI_API_KEY=\S+' .env; then
  info ".env già presente con una chiave (non viene mostrata né modificata)"
else
  info "La chiave serve solo per il resoconto scritto dall'IA e per il tutor (costo tipico: < 1 centesimo a brano)."
  info "Senza chiave la tablatura funziona lo stesso. La chiave resta solo in .env (escluso da git)."
  read -r -s -p "    Incolla la TUA OPENAI_API_KEY (invio per saltare): " CHIAVE; echo
  CHIAVE="${CHIAVE//[[:space:]]/}"
  sed "s|^OPENAI_API_KEY=.*$|OPENAI_API_KEY=${CHIAVE}|" .env.esempio > .env
  chmod 600 .env
  unset CHIAVE
  if grep -qE '^OPENAI_API_KEY=\S+' .env; then info ".env creato con la chiave"; else info ".env creato senza chiave"; fi
fi

# ---------------------------------------------------------------- 8. modelli, suoni, indice del tutor
passo "8/9" "Modelli, banco di suoni e indice del tutor"
OPZ=()
[[ $SPERIMENTALI -eq 1 ]] && OPZ+=(--sperimentali)
[[ $SENZA_MODELLI -eq 1 ]] && OPZ+=(--senza-modelli)
"$VENV_PY" scripts/prepara_risorse.py "${OPZ[@]+"${OPZ[@]}"}" || info "alcune risorse non sono state preparate: arriveranno al primo uso."

# ---------------------------------------------------------------- 9. verifica
passo "9/9" "Verifica dell'ambiente"
OK=1
"$VENV_PY" scripts/check_setup.py || OK=0
if [[ $SENZA_TEST -eq 0 ]]; then
  info "test automatici (1-3 minuti; i test sui brani reali vengono saltati se mancano gli MP3)..."
  "$VENV_PY" -m pytest -q || OK=0
fi

MIN=$(( ($(date +%s) - INIZIO) / 60 ))
if [[ $OK -eq 1 ]]; then printf '\n\033[32mPRONTO (%s min).\033[0m\n' "$MIN"
else printf '\n\033[33mINIZIALIZZAZIONE COMPLETATA CON PROBLEMI (%s min): vedi i [KO] sopra.\033[0m\n' "$MIN"; fi
cat <<'FINE'

Avvio:
  pagina web       .venv/bin/python -m app.web.server        poi apri http://127.0.0.1:8000
  riga di comando  .venv/bin/python -m app.orchestratore brano.mp3
  server MCP       .venv/bin/python -m app.mcp_server        (vedi README per Claude Desktop)
FINE
