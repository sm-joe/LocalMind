<#
.SYNOPSIS
  Download an Ollama model using temporary outbound network access.

.DESCRIPTION
  LocalRAG runs Ollama on an internal-only Docker network. This script
  temporarily attaches its running container to the Compose egress network,
  pulls the model, and disconnects it in finally. Existing volumes are not
  modified or deleted.
.EXAMPLE
  .\scripts\pull-ollama-model.ps1 -Model qwen3:8b
#>
param(
    [Parameter(Mandatory = $false)]
    [ValidateNotNullOrEmpty()]
    [string]$Model = 'qwen3:8b'
)

$ErrorActionPreference = 'Stop'

$networks = @(
    docker network ls --format '{{.Name}}' |
        Where-Object { $_ -like '*ollama-egress' }
)
if ($LASTEXITCODE -ne 0) {
    throw 'Could not list Docker networks.'
}
if ($networks.Count -ne 1) {
    throw "Expected exactly one LocalRAG ollama-egress network; found $($networks.Count)."
}

$container = 'localrag-ollama'
$network = $networks[0]
$currentNetworks = @(
    docker inspect --format '{{range $name, $network := .NetworkSettings.Networks}}{{println $name}}{{end}}' $container
)
if ($LASTEXITCODE -ne 0) {
    throw 'Running LocalRAG Ollama container not found. Start it with docker compose up -d ollama.'
}
$alreadyConnected = $network -in $currentNetworks
$addedHere = $false

try {
    if (-not $alreadyConnected) {
        docker network connect $network $container
        if ($LASTEXITCODE -ne 0) {
            throw "Could not connect Ollama to egress network $network."
        }
        $addedHere = $true
    }
    docker compose exec -T ollama ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama model download failed: $Model."
    }
    Write-Host "Model $Model downloaded successfully." -ForegroundColor Green
}
finally {
    if ($addedHere) {
        docker network disconnect $network $container
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Disconnect Ollama from $network manually before continuing."
        }
    }
}
