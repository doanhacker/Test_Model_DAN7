param([switch]$SkipDocker)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $SkipDocker) {
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw 'Không khởi động được Elasticsearch/Qdrant. Hãy mở Docker Desktop hoặc dùng -SkipDocker với dịch vụ đã chạy.' }
}
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
