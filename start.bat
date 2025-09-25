@echo off
chcp 65001 >nul
title LLM Chat (Docker)
echo ========================================
echo Запуск LLM Chat в Docker
echo ========================================
echo.

REM Имена и сеть
set NET_NAME=chat_net
set QDRANT_IMAGE=qdrant/qdrant
set QDRANT_NAME=qdrant
set APP_IMAGE=chat-app-image
set APP_NAME=chat-app

REM Env по умолчанию
if "%OLLAMA_URL%"=="" set OLLAMA_URL=http://host.docker.internal:11434
if "%OLLAMA_EMBED_MODEL%"=="" set OLLAMA_EMBED_MODEL=all-minilm

REM Проверка Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Docker не найден!
    pause
    exit /b 1
)

REM Сеть
docker network inspect %NET_NAME% >nul 2>&1 || docker network create %NET_NAME% >nul

REM Qdrant образ
docker image inspect %QDRANT_IMAGE% >nul 2>&1 || docker pull %QDRANT_IMAGE%

REM Поднимаем Qdrant (в сети), монтируем storage
docker ps -a --format "{{.Names}}" | findstr /I ^%QDRANT_NAME%$ >nul
if %errorlevel% neq 0 (
    echo Создаем контейнер Qdrant...
    docker run -d --name %QDRANT_NAME% --network %NET_NAME% -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage":/qdrant/storage %QDRANT_IMAGE%
) else (
    echo Запускаем Qdrant...
    docker start %QDRANT_NAME% >nul
)

REM Ждем готовности Qdrant по логам (гарантированно в сети Docker)
echo Ждем готовности Qdrant (gRPC 6334)...
set _tries=0
:wait_q
docker logs --since 1s %QDRANT_NAME% 2>nul | findstr /C:"gRPC listening on 6334" >nul
if %errorlevel% neq 0 (
  set /a _tries+=1
  if %_tries% gtr 60 (
    echo Не дождались gRPC 6334. Последние логи:
    docker logs --tail 80 %QDRANT_NAME%
    goto end
  )
  timeout /t 1 >nul
  goto wait_q
)
echo Qdrant готов к gRPC.

REM Проверяем образ приложения (если нет — собираем)
docker image inspect %APP_IMAGE% >nul 2>&1
if %errorlevel% neq 0 (
    echo Образ %APP_IMAGE% не найден. Собираем...
    docker build -t %APP_IMAGE% . || goto :end
)

REM Старт приложения в той же сети и с Qdrant host=container
echo Запускаем приложение...
2>nul docker rm -f %APP_NAME% >nul
docker run --name %APP_NAME% --network %NET_NAME% --rm -p 8501:8501 ^
  -e QDRANT_HOST=%QDRANT_NAME% ^
  -e OLLAMA_URL=%OLLAMA_URL% ^
  -e OLLAMA_EMBED_MODEL=%OLLAMA_EMBED_MODEL% ^
  %APP_IMAGE%

echo Приложение завершилось. Останавливаем Qdrant...
docker stop %QDRANT_NAME% >nul

:end
pause