@echo off
chcp 65001 >nul
title Установка зависимостей LLM Chat

echo ========================================
echo Установка зависимостей для LLM Chat
echo ========================================
echo.

echo 📦 Установка основных зависимостей...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ❌ Ошибка установки основных зависимостей!
    goto end
)

echo.
echo ✅ Основные зависимости установлены успешно!

echo.
echo 📎 Установка опциональных зависимостей...
pip install -r requirements_optional.txt

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Ошибка установки опциональных зависимостей (это не критично)
) else (
    echo ✅ Опциональные зависимости установлены успешно!
)

echo.
echo 🎉 Все зависимости обработаны!

:end
echo.
echo Нажмите любую клавишу для выхода...
pause >nul