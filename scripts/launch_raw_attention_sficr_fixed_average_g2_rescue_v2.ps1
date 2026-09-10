param(
    [string]$RawDir = "data/local/open",
    [string]$OutDir = "artifacts\postgate\raw_attention_sficr_fixed_average_g2_rescue_v2"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "scripts\run_raw_attention_sficr_fixed_average_g2_rescue.py"
$Config = Join-Path $ProjectRoot "configs\raw_attention_sficr_fixed_average_g2_rescue_preregister_v2.json"
$ResolvedOut = Join-Path $ProjectRoot $OutDir
$LogDir = Join-Path $ProjectRoot "artifacts\logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Stdout = Join-Path $LogDir "raw_attention_sficr_fixed_average_g2_rescue_v2.stdout.log"
$Stderr = Join-Path $LogDir "raw_attention_sficr_fixed_average_g2_rescue_v2.stderr.log"

if (Test-Path -LiteralPath $ResolvedOut) {
    throw "Fresh canonical already exists: $ResolvedOut"
}

$Arguments = @(
    "-B", $Runner,
    "--stage", "all",
    "--raw-dir", $RawDir,
    "--out-dir", $ResolvedOut,
    "--config", $Config
)
$Process = Start-Process -FilePath $PythonExe -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
[pscustomobject]@{
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    canonical = $ResolvedOut
}
