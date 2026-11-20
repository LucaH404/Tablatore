<#
.SYNOPSIS
  Inizializza il Tablatore AI su Windows: ambiente Python 3.11, dipendenze, ffmpeg, .env, modelli, verifica.

.DESCRIPTION
  Da lanciare nella cartella del progetto:
      powershell -ExecutionPolicy Bypass -File scripts\inizializza.ps1
  Si può rilanciare: i passi già fatti vengono saltati o ripetuti senza danni.
  Serve solo un Python di sistema (3.8 o più recente) per installare uv; uv scarica da solo Python 3.11.
  Download: ~4-5 GB con GPU NVIDIA (PyTorch CUDA), ~1 GB senza; modelli ~0,4 GB; banco di suoni 31 MB.

.PARAMETER Cpu           Installa PyTorch per CPU anche se c'è una GPU NVIDIA.
.PARAMETER Sperimentali  Installa anche il separatore sperimentale BS-RoFormer (+700 MB di pesi).
.PARAMETER SenzaModelli  Non scarica in anticipo i modelli (arriveranno al primo uso).
.PARAMETER SenzaTest     Salta la verifica finale con pytest.
#>
param([switch]$Cpu, [switch]$Sperimentali, [switch]$SenzaModelli, [switch]$SenzaTest)

$ErrorActionPreference = "Continue"      # i programmi esterni scrivono l'avanzamento su stderr: si controlla $LASTEXITCODE
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$TorchVer = "2.14.1"
$inizio = Get-Date

function Passo([string]$n, [string]$testo) { Write-Host "`n[$n] $testo" -ForegroundColor Cyan }
function Info([string]$testo) { Write-Host "    $testo" }
function Fallisci([string]$testo) { Write-Host "`nERRORE: $testo" -ForegroundColor Red; exit 1 }
function Esegui([string]$descrizione, [scriptblock]$comando) {
    & $comando
    if ($LASTEXITCODE -ne 0) { Fallisci "$descrizione (codice $LASTEXITCODE)" }
}
function Aggiorna-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

Write-Host "Tablatore AI - inizializzazione ($Root)" -ForegroundColor Green

# ---------------------------------------------------------------- 1. Python di sistema e uv
Passo "1/9" "Python di sistema e uv"
$SysPy = $null
foreach ($candidato in @(@("py", "-3"), @("python"), @("python3"))) {
    $exe = $candidato[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $argomenti = @($candidato | Select-Object -Skip 1) + @("-c", "import sys; print(sys.executable) if sys.version_info >= (3, 8) else exit(1)")
    $percorso = & $exe @argomenti 2>$null
    if ($LASTEXITCODE -eq 0 -and $percorso -and ($percorso -notmatch "WindowsApps")) { $SysPy = "$percorso".Trim(); break }
}
if (-not $SysPy) { Fallisci "nessun Python 3.8+ trovato. Installalo da https://www.python.org/downloads/ (spunta 'Add to PATH') e rilancia." }
Info "Python di sistema: $SysPy"
& $SysPy -m uv --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Info "installo uv (gestore di pacchetti) per l'utente corrente..."
    Esegui "installazione di uv" { & $SysPy -m pip install --user --upgrade uv }
}
Info ("uv: " + (& $SysPy -m uv --version))
# uv va chiamato dal Python di sistema, non da quello del .venv
function Uv { & $SysPy -m uv @args }

# ---------------------------------------------------------------- 2. ambiente Python 3.11
Passo "2/9" "Ambiente virtuale .venv con Python 3.11"
if (Test-Path $VenvPy) {
    $ver = & $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($ver -ne "3.11") { Fallisci ".venv esiste ma usa Python ${ver}: rinominala o cancellala e rilancia." }
    Info ".venv già presente (Python 3.11)"
} else {
    Esegui "creazione del .venv" { Uv venv --python 3.11 .venv }     # uv scarica Python 3.11 se manca
}

# ---------------------------------------------------------------- 3. PyTorch
Passo "3/9" "PyTorch $TorchVer"
$gpu = (-not $Cpu) -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$indice = if ($gpu) { "cu126" } else { "cpu" }
if ($gpu) { Info "GPU NVIDIA trovata: versione CUDA 12.6 (download ~4-5 GB, può richiedere parecchi minuti)" }
else { Info "nessuna GPU NVIDIA (o -Cpu): versione per CPU. La separazione sarà ~10 volte più lenta." }
Esegui "installazione di PyTorch" { Uv pip install --python $VenvPy "torch==$TorchVer" --index-url "https://download.pytorch.org/whl/$indice" }

# ---------------------------------------------------------------- 4. pacchetti senza dipendenze
Passo "4/9" "Basic Pitch, beat_this, torchaudio (senza dipendenze)"
# basic-pitch con le dipendenze installerebbe TensorFlow; torchaudio/beat-this cambierebbero torch
Esegui "pacchetti senza dipendenze" { Uv pip install --python $VenvPy --no-deps "basic-pitch==0.4.0" "beat-this==1.1.0" "torchaudio==2.11.1" "rotary-embedding-torch==0.9.2" }

# ---------------------------------------------------------------- 5. requirements.txt
Passo "5/9" "Dipendenze del progetto (requirements.txt)"
Esegui "requirements.txt" { Uv pip install --python $VenvPy -r requirements.txt }
if ($Sperimentali) {
    Info "separatore sperimentale BS-RoFormer..."
    # NON usare audio-separator: declassa rotary-embedding-torch e rompe beat_this
    Esegui "bs-roformer-infer" { Uv pip install --python $VenvPy "bs-roformer-infer==0.1.5" }
}

# ---------------------------------------------------------------- 6. ffmpeg
Passo "6/9" "ffmpeg"
Aggiorna-Path
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Info "ffmpeg già presente: $((Get-Command ffmpeg).Source)"
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    $risposta = Read-Host "    ffmpeg manca. Lo installo con winget (Gyan.FFmpeg)? [S/n]"
    if ($risposta -eq "" -or $risposta -match "^[sSyY]") {
        Esegui "installazione di ffmpeg" { winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements }
        Aggiorna-Path
        if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Info "ffmpeg installato: apri un nuovo terminale se non viene trovato." }
    } else { Info "ffmpeg saltato: l'app non funziona senza (https://ffmpeg.org/download.html)." }
} else {
    Info "ffmpeg manca e winget non c'è: installalo da https://ffmpeg.org/download.html e aggiungilo al PATH."
}

# ---------------------------------------------------------------- 7. .env con la chiave OpenAI
Passo "7/9" "File .env (chiave OpenAI)"
$fileEnv = Join-Path $Root ".env"
$chiavePresente = (Test-Path $fileEnv) -and (Select-String -Path $fileEnv -Pattern "^OPENAI_API_KEY=\S+" -Quiet)
if ($chiavePresente) {
    Info ".env già presente con una chiave (non viene mostrata né modificata)"
} else {
    Info "La chiave serve solo per il resoconto scritto dall'IA e per il tutor (costo tipico: < 1 centesimo a brano)."
    Info "Senza chiave la tablatura funziona lo stesso. La chiave resta solo in .env (escluso da git)."
    $sicura = Read-Host "    Incolla la TUA OPENAI_API_KEY (invio per saltare)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sicura)
    $chiave = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $contenuto = (Get-Content (Join-Path $Root ".env.esempio") -Raw) -replace "(?m)^OPENAI_API_KEY=.*$", "OPENAI_API_KEY=$chiave"
    [IO.File]::WriteAllText($fileEnv, $contenuto, (New-Object System.Text.UTF8Encoding $false))
    $chiave = $null
    if ($contenuto -match "(?m)^OPENAI_API_KEY=\S+") { Info ".env creato con la chiave" } else { Info ".env creato senza chiave (si può aggiungere dopo)" }
}

# ---------------------------------------------------------------- 8. modelli, suoni, indice del tutor
Passo "8/9" "Modelli, banco di suoni e indice del tutor"
$opzioni = @()
if ($Sperimentali) { $opzioni += "--sperimentali" }
if ($SenzaModelli) { $opzioni += "--senza-modelli" }
& $VenvPy scripts\prepara_risorse.py @opzioni
if ($LASTEXITCODE -ne 0) { Info "alcune risorse non sono state preparate (vedi sopra): arriveranno al primo uso." }

# ---------------------------------------------------------------- 9. verifica
Passo "9/9" "Verifica dell'ambiente"
& $VenvPy scripts\check_setup.py
$ambienteOk = ($LASTEXITCODE -eq 0)
if (-not $SenzaTest) {
    Info "test automatici (1-3 minuti; i test sui brani reali vengono saltati se mancano gli MP3)..."
    & $VenvPy -m pytest -q
    if ($LASTEXITCODE -ne 0) { $ambienteOk = $false }
}

$minuti = [math]::Round(((Get-Date) - $inizio).TotalMinutes, 1)
if ($ambienteOk) { Write-Host "`nPRONTO ($minuti min)." -ForegroundColor Green }
else { Write-Host "`nINIZIALIZZAZIONE COMPLETATA CON PROBLEMI ($minuti min): vedi i [KO] sopra." -ForegroundColor Yellow }
Write-Host @"

Avvio:
  pagina web     .venv\Scripts\python.exe -m app.web.server        poi apri http://127.0.0.1:8000
  riga di comando .venv\Scripts\python.exe -m app.orchestratore brano.mp3
  server MCP     .venv\Scripts\python.exe -m app.mcp_server         (vedi README per Claude Desktop)
"@
