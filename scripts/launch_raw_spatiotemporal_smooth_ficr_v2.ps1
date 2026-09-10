Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$guardPath = Join-Path $projectRoot "artifacts\locks\heavy_cpu_fit.pid.json"
$canonicalPath = Join-Path $projectRoot "artifacts\postgate\raw_spatiotemporal_smooth_ficr_strict_v2"
$runlogDirectory = Join-Path $projectRoot "artifacts\runlogs"
$stdoutPath = Join-Path $runlogDirectory "raw_spatiotemporal_smooth_ficr_strict_v2.stdout.log"
$stderrPath = Join-Path $runlogDirectory "raw_spatiotemporal_smooth_ficr_strict_v2.stderr.log"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "registered venv Python is absent: $pythonPath"
}
if (Test-Path -LiteralPath $canonicalPath) {
    throw "fresh canonical already exists: $canonicalPath"
}
if (Test-Path -LiteralPath $guardPath) {
    throw "shared heavy guard already exists: $guardPath"
}
if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath)) {
    throw "fresh stdout/stderr path already exists"
}

$existingPython = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match '^python(w)?\.exe$' }
)
if ($existingPython.Count -ne 0) {
    $ids = ($existingPython | ForEach-Object { $_.ProcessId }) -join ","
    throw "Python PID count must be zero before launch; observed: $ids"
}

New-Item -ItemType Directory -Path $runlogDirectory -Force | Out-Null
$arguments = @(
    "-B",
    "scripts\run_raw_spatiotemporal_smooth_ficr.py",
    "--stage", "all",
    "--config", "configs\raw_spatiotemporal_smooth_ficr_preregister_v2.json",
    "--out-dir", "artifacts\postgate\raw_spatiotemporal_smooth_ficr_strict_v2"
)
$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$guard = $null
for ($attempt = 0; $attempt -lt 50; $attempt++) {
    if (Test-Path -LiteralPath $guardPath -PathType Leaf) {
        try {
            $guard = Get-Content -Raw -LiteralPath $guardPath | ConvertFrom-Json
            if ($guard.experiment_id -eq "raw_spatiotemporal_smooth_ficr_strict_forward_v2") {
                break
            }
        } catch {
            $guard = $null
        }
    }
    if ($process.HasExited) {
        throw "launched venv shim exited before acquiring the registered guard"
    }
    Start-Sleep -Milliseconds 200
}
if ($null -eq $guard) {
    throw "registered worker did not acquire the shared guard within 10 seconds"
}

$matching = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python(w)?\.exe$' -and
            $_.CommandLine -like '*run_raw_spatiotemporal_smooth_ficr.py*'
        }
)
if ($matching.Count -lt 1 -or $matching.Count -gt 2) {
    throw "expected one venv shim/worker process tree; observed $($matching.Count) matching processes"
}
$allowedIds = @([int]$process.Id, [int]$guard.pid)
foreach ($item in $matching) {
    if ($allowedIds -notcontains [int]$item.ProcessId) {
        throw "matching Python process is outside the registered single tree: $($item.ProcessId)"
    }
}
if ($allowedIds -notcontains [int]$guard.pid) {
    throw "guard owner is outside the launched process tree"
}

[ordered]@{
    schema_version = 1
    launcher_pid = [int]$PID
    venv_shim_pid = [int]$process.Id
    guard_worker_pid = [int]$guard.pid
    guard_token = [string]$guard.token
    matching_process_count = [int]$matching.Count
    matching_processes = @(
        $matching | ForEach-Object {
            [ordered]@{
                pid = [int]$_.ProcessId
                parent_pid = [int]$_.ParentProcessId
                name = [string]$_.Name
                command_line = [string]$_.CommandLine
            }
        }
    )
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    canonical_path = $canonicalPath
    hidden_window = $true
    durable_redirects = $true
} | ConvertTo-Json -Depth 6 -Compress
