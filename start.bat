@echo off
chcp 65001 >nul
title LLM Chat Application
echo ========================================
echo Запуск LLM Chat приложения
echo ========================================
echo.

REM Проверка Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Docker не найден!
    pause
    exit /b 1
)

REM Проверка наличия образа Qdrant
docker image inspect qdrant/qdrant >nul 2>&1
if %errorlevel% neq 0 (
    echo Образ Qdrant не найден, скачиваем...
    docker pull qdrant/qdrant
)

REM Проверка наличия контейнера Qdrant
docker ps -a --format "{{.Names}}" | findstr qdrant_container >nul
if %errorlevel% neq 0 (
    echo Контейнер Qdrant не найден, создаем...
    docker run -d --name qdrant_container -p 6333:6333 -p 6334:6334 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant
) else (
    echo Контейнер Qdrant найден, запускаем если остановлен...
    docker start qdrant_container
)

REM Ждем доступности gRPC порта 6334
echo Ждем, пока Qdrant gRPC порт 6334 станет доступен...
:wait_grpc
powershell -Command "try { $tcp=Test-NetConnection -ComputerName 127.0.0.1 -Port 6334; if($tcp.TcpTestSucceeded){exit 0}else{exit 1} } catch {exit 1}"
if %errorlevel% neq 0 (
    timeout /t 1 >nul
    goto wait_grpc
)
echo gRPC Qdrant готов!

REM Проверка Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Python не найден!
    pause
    exit /b 1
)

REM Запуск Streamlit приложения - ВАЖНО: теперь main.py в папке app
echo Запуск приложения Streamlit...
echo Откройте браузер по адресу: http://localhost:8501
cd app
streamlit run main.py --server.port 8501
cd ..

REM После закрытия Streamlit останавливаем контейнер
echo Streamlit завершен, останавливаем Qdrant контейнер...
docker stop qdrant_container

pause