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
function Remove-Stale {
    $project = "fraud-pipeline"
    $rows = docker ps -a --filter "name=fraud-" --format '{{.Names}}|{{.Labels}}' 2>$null
    foreach ($r in $rows) {
        if (-not $r) { continue }
        if ($r -notmatch "com\.docker\.compose\.project=$project(,|$)") {
            $name = $r.Split('|')[0]
            Info "Removing leftover container from another project: $name"
            docker rm -f $name *> $null
        }
    }
    $net = docker network inspect fraud-net --format '{{.Labels}}' 2>$null
    if ($LASTEXITCODE -eq 0 -and ($net | Out-String) -notmatch "com\.docker\.compose\.project:$project") {
        Info "Removing leftover network fraud-net"
        docker network rm fraud-net *> $null
    }
    foreach ($v in @("fraud-cassandra-data", "fraud-grafana-data", "fraud-prometheus-data", "fraud-kafka-data")) {
        $lab = docker volume inspect $v --format '{{.Labels}}' 2>$null
        if ($LASTEXITCODE -eq 0 -and ($lab | Out-String) -notmatch "com\.docker\.compose\.project:$project") {
            Info "Removing leftover volume $v (old demo data)"
            docker volume rm $v *> $null
        }
    }
}

function Cmd-Urls {
    Write-Host @"

Dashboard (alerts)  http://localhost:8501
Grafana (metrics)   http://localhost:3000   (admin / admin)
Kafka UI            http://localhost:8080
Prometheus          http://localhost:9090
Scorer metrics      http://localhost:8000/metrics
"@
}

function Cmd-Up {
    Assert-Docker; Ensure-Env
    Remove-Stale
    docker compose config -q
    if ($LASTEXITCODE -ne 0) { Fail "docker-compose.yml is invalid." }
    Info "Starting infrastructure (first run downloads images - be patient)..."
    docker compose up -d --remove-orphans kafka cassandra prometheus grafana kafka-ui kafka-exporter
    if ($LASTEXITCODE -ne 0) {
        Info "Start failed; retrying once from a clean state..."
        docker compose down --remove-orphans *> $null
        docker compose up -d --remove-orphans kafka cassandra prometheus grafana kafka-ui kafka-exporter
        if ($LASTEXITCODE -ne 0) { Fail "Still failing. Run: docker compose logs --tail 80" }
    }
    $deadline = (Get-Date).AddMinutes(6)
    while ((Get-Date) -lt $deadline) {
        $pending = @()
        foreach ($n in @("fraud-kafka", "fraud-cassandra", "fraud-prometheus", "fraud-grafana")) {
            $h = Get-Health $n
            if ($h -ne "healthy") { $pending += "$n=$h" }
        }
        if ($pending.Count -eq 0) { Ok "Infrastructure is healthy."; return }
        Write-Host ("  waiting: " + ($pending -join ", "))
        Start-Sleep -Seconds 8
    }
    Fail "Timed out waiting for services. Try: .\run.ps1 logs cassandra"
}

function Cmd-Data  { Assert-Docker; Ensure-Env; Info "Generating data..."; docker compose run --rm datagen; if ($LASTEXITCODE -ne 0) { Fail "data step failed" } }
function Cmd-Train { Assert-Docker; Ensure-Env; Info "Training model..."; docker compose run --rm trainer; if ($LASTEXITCODE -ne 0) { Fail "training failed" } }

function Cmd-Start {
    Assert-Docker; Ensure-Env
    if (-not (Test-Path "models\isolation_forest.joblib")) { Fail "No model yet. Run: .\run.ps1 data ; .\run.ps1 train" }
    Info "Starting producer, scorer, dashboard..."
    docker compose --profile app up -d --build scorer producer dashboard
    if ($LASTEXITCODE -ne 0) { Fail "could not start app services" }
    Ok "App started. Give it ~30s to connect to Cassandra and Kafka, then run: .\run.ps1 health"
    Cmd-Urls
}

function Cmd-Health {
    Assert-Docker
    $fail = 0
    foreach ($n in @("fraud-kafka", "fraud-cassandra", "fraud-prometheus", "fraud-grafana")) {
        $h = Get-Health $n
        if ($h -eq "healthy") { Ok "$n" } else { Bad "$n ($h)"; $fail++ }
    }
    foreach ($n in @("fraud-scorer", "fraud-producer", "fraud-dashboard")) {
        $h = Get-Health $n
        if ($h -eq "running") { Ok "$n" } else { Bad "$n ($h)"; $fail++ }
    }
    if (Test-Http "http://localhost:3000/api/health") { Ok "Grafana API" } else { Bad "Grafana API"; $fail++ }
    if (Test-Http "http://localhost:9090/-/healthy")  { Ok "Prometheus API" } else { Bad "Prometheus API"; $fail++ }
    if (Test-Http "http://localhost:8080")            { Ok "Kafka UI" } else { Bad "Kafka UI"; $fail++ }
    if (Test-Http "http://localhost:8501")            { Ok "Dashboard" } else { Bad "Dashboard"; $fail++ }
    try {
        $m = (Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing -TimeoutSec 5).Content
        $line = ($m -split "`n" | Where-Object { $_ -match "^fraud_transactions_processed_total " }) | Select-Object -First 1
        Ok "Scorer metrics: $line"
    } catch { Bad "Scorer metrics (is the scorer running?)"; $fail++ }
    if ($fail -eq 0) { Write-Host "`nAll checks passed." -ForegroundColor Green } else { Write-Host "`n$fail check(s) failed. See: .\run.ps1 logs <service>" -ForegroundColor Red }
}

