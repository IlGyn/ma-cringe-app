@echo off
setlocal ENABLEDELAYEDEXPANSION

REM -----------------------------
REM Defaults (can be overridden via env)
REM -----------------------------
if "%OLLAMA_URL%"=="" set OLLAMA_URL=http://host.docker.internal:11434
if "%OLLAMA_EMBED_MODEL%"=="" set OLLAMA_EMBED_MODEL=all-minilm
if "%QDRANT_HOST%"=="" set QDRANT_HOST=qdrant
if "%QDRANT_PORT%"=="" set QDRANT_PORT=6334
if "%PORT%"=="" set PORT=8501

REM -----------------------------
REM Usage: run_docker.bat [build|up|down|restart] [port]
REM -----------------------------
set ACTION=%1%
if "%ACTION%"=="" set ACTION=up

REM -----------------------------
REM Build images
REM -----------------------------
if /I "%ACTION%"=="build" (
    echo [BUILD] Building chat-app image ...
    docker build -t chat-app-image .
    if errorlevel 1 exit /b 1
    echo [BUILD] Done.
    exit /b 0
)

REM -----------------------------
REM Start services via docker-compose
REM -----------------------------
if /I "%ACTION%"=="up" (
    echo [UP] Starting services...
    docker-compose up -d --build
    exit /b %errorlevel%
)

REM -----------------------------
REM Stop services via docker-compose
REM -----------------------------
if /I "%ACTION%"=="down" (
    echo [DOWN] Stopping services...
    docker-compose down
    exit /b %errorlevel%
)

REM -----------------------------
REM Restart services
REM -----------------------------
if /I "%ACTION%"=="restart" (
    echo [RESTART] Restarting services...
    docker-compose down
    docker-compose up -d --build
    exit /b %errorlevel%
)

echo Usage: %~n0 [build^|up^|down^|restart] [port]
echo    OLLAMA_URL default: %OLLAMA_URL%
echo    OLLAMA_EMBED_MODEL default: %OLLAMA_EMBED_MODEL%
echo    QDRANT_HOST default: %QDRANT_HOST%
echo    QDRANT_PORT default: %QDRANT_PORT%
exit /b 1
