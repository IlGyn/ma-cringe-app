@echo off
chcp 65001 >nul
title Проверка зависимостей LLM Chat

echo ========================================
echo Проверка зависимостей для LLM Chat
echo ========================================
echo.

python check_deps.py
set CHECK_RESULT=%errorlevel%

if %CHECK_RESULT% == 0 (
    echo.
    echo ✅ Все основные зависимости установлены!
    echo.
    echo 🚀 Можно запускать приложение:
    echo    start.bat
    goto end
)

echo.
echo ⚠️  Обнаружены отсутствующие зависимости.
echo Хотите установить их сейчас? (Y/N)
set /p USER_CHOICE=""

if /i "%USER_CHOICE%"=="Y" (
    echo.
    echo 🚀 Установка отсутствующих зависимостей...
    call install_deps.bat
    echo.
    echo 🔄 Перепроверка зависимостей...
    python check_deps.py
) else (
    echo.
    echo ℹ️  Установка отменена.
    echo Запустите install_deps.bat вручную при необходимости.
)

:end
echo.
echo Нажмите любую клавишу для выхода...
pause >nul