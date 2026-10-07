<#
 run.ps1 - one-command runner for Windows PowerShell (no make/bash needed).

   .\run.ps1 all        # everything: infra -> data -> train -> app   (first run: ~5-10 min)
   .\run.ps1 up         # start Kafka, Cassandra, Prometheus, Grafana, Kafka UI
   .\run.ps1 data       # generate synthetic PaySim-style data (skipped if data\paysim.csv exists)
   .\run.ps1 train      # train the Isolation Forest, write docs\model_evaluation.md
   .\run.ps1 start      # start producer + scorer + dashboard
   .\run.ps1 health     # check every component
   .\run.ps1 urls       # print service URLs
   .\run.ps1 logs scorer
   .\run.ps1 test       # run unit tests in a container
   .\run.ps1 stop       # stop producer/scorer/dashboard only
   .\run.ps1 down       # stop everything, keep data
   .\run.ps1 clean      # stop everything and delete all data
#>
param(
    [Parameter(Position = 0)][string]$Command = "help",
    [Parameter(Position = 1)][string]$Service = ""
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Ok($m)   { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Bad($m)  { Write-Host "[FAIL] $m" -ForegroundColor Red }
function Info($m) { Write-Host ">> $m" -ForegroundColor Cyan }

function Assert-Docker {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { Fail "Docker is not running. Start Docker Desktop, wait for 'running', then retry." }
}

function Ensure-Env {
    if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Ok ".env created" }
}

function Get-Health($name) {
    $r = docker inspect -f "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" $name 2>$null
    if ($LASTEXITCODE -ne 0) { return "missing" }
    return ($r | Out-String).Trim()
}

function Test-Http($url) {
    try { $null = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5; return $true } catch { return $false }
}


# Remove leftovers from an older/other copy of this project (same fixed names, different compose project).
