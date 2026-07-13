<#
.SYNOPSIS
    Collect the Android app's per-example bench.json into a local output root
    (Windows/PowerShell port of pull_android_benches.sh).

.DESCRIPTION
    On Android the examples run inside the app (org.dressi.examples) driven by
    its UI -- there is NO headless adb entry point, so this script does NOT
    start runs. Run each example on-device from the app first (they write
    files/out/<example>/bench.json), then pull the records here for
    aggregation:

        scripts\pull_android_benches.ps1 -OutRoot bench_results\android_pixel8
        python scripts\bench_summary.py bench_results\android_pixel8

    Files are read out of the app's private storage with `run-as`, so a
    debuggable build is required (the assembleDebug APK is; the release
    benchmarking build is configured debuggable too -- see README-android.md).

    Run this on the HOST PC (it drives the device over USB via adb), not on
    the phone.

.PARAMETER OutRoot
    Local output root (default: bench_results\android).
.PARAMETER Serial
    adb device serial (default: the only connected device).
.PARAMETER Package
    App package (default: org.dressi.examples).
.PARAMETER Only
    Comma-separated example-name substrings; only matching ones are pulled.
.PARAMETER List
    List the on-device example out-dir names and exit.
#>
[CmdletBinding()]
param(
    [string]$OutRoot = 'bench_results/android',
    [string]$Serial  = '',
    [string]$Package = 'org.dressi.examples',
    [string]$Only    = '',
    [switch]$List
)

$ErrorActionPreference = 'Stop'

# On-device out/<dir> names. These equal the registry example names EXCEPT
# silhouette, which the app runs as two technique variants each into its own
# out-dir (out/silhouette_optimization_<technique>); see
# examples/common/example_registry.cpp and MainActivity.defaultArgs.
$examples = @(
    'image_fitting'
    'texture_optimization'
    'silhouette_optimization_hardsoftras'
    'silhouette_optimization_aa'
    'pbr_shading'
    'pbr_material_optimization'
    'pbr_envmap_optimization'
)

function Test-Match([string]$name) {
    if ([string]::IsNullOrEmpty($Only)) { return $true }
    foreach ($pat in $Only.Split(',')) {
        if ($pat -and $name.Contains($pat)) { return $true }
    }
    return $false
}

if ($List) {
    $examples | ForEach-Object { $_ }
    exit 0
}

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Write-Error 'adb not found on PATH'
    exit 1
}

# Build the adb invocation prefix (optionally with -s serial).
$adbArgs = @()
if ($Serial) { $adbArgs += @('-s', $Serial) }

# Fail fast if no device is reachable.
& adb @adbArgs get-state *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "no adb device (check 'adb devices' / USB debugging / -Serial)"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$fail = 0
$summary = [System.Collections.Generic.List[object]]::new()
foreach ($name in $examples) {
    if (-not (Test-Match $name)) { continue }

    $remote   = "files/out/$name/bench.json"
    $localDir = Join-Path $OutRoot $name
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null

    # run-as streams the file out of the app's private storage. Empty output
    # (or non-zero exit) == not produced yet. Strip CRs adb may inject.
    $json = (& adb @adbArgs shell run-as $Package cat $remote 2>$null) -join "`n"
    $json = $json -replace "`r", ''
    if ($LASTEXITCODE -eq 0 -and $json.Trim()) {
        $path = Join-Path $localDir 'bench.json'
        [System.IO.File]::WriteAllText(
            $path, $json, (New-Object System.Text.UTF8Encoding($false)))
        $summary.Add([pscustomobject]@{ example = $name; status = 'pulled' })
    } else {
        Write-Warning "${name}: no $remote (run it in the app first?)"
        $summary.Add([pscustomobject]@{ example = $name; status = 'MISSING' })
        $fail = 1
    }
}

Write-Host ''
Write-Host '=== android pull summary ==='
$summary | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "Aggregate with:  python scripts\bench_summary.py $OutRoot"

exit $fail
