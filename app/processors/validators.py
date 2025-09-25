import html
import logging
import streamlit as st
from urllib.parse import urlparse
from ..config import Config

# Проверка доступности опциональных библиотек
try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import openpyxl
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

class Validator:
    def validate_prompt(self, prompt: str) -> bool:
        if not prompt or not prompt.strip():
            st.error("Пожалуйста, введите непустое сообщение")
            logging.warning("Попытка отправки пустого сообщения")
            return False
        
        if len(prompt) > Config.MAX_MESSAGE_LENGTH:
            st.error(f"Сообщение слишком длинное (максимум {Config.MAX_MESSAGE_LENGTH} символов)")
            return False

        # Расширенная проверка безопасности
        dangerous_patterns = [
            "<?php", "<script", "javascript:", "onload=", 
            "base64_decode", "eval(", "system(", "exec(",
            "union select", "drop table", "insert into",
            "delete from", "update.*set", "create table"
        ]
        
        # HTML escape для дополнительной безопасности
        prompt_sanitized = html.escape(prompt.lower())
        
        if any(pattern in prompt_sanitized for pattern in dangerous_patterns):
            st.error("Сообщение содержит подозрительные паттерны")
            logging.warning(f"Обнаружены опасные паттерны в сообщении: {prompt[:100]}...")
            return False
            
        # Дополнительная проверка на SQL-инъекции
        sql_keywords = ['select', 'insert', 'update', 'delete', 'drop', 'create', 'alter']
        if any(keyword in prompt_sanitized for keyword in sql_keywords):
            # Проверяем на подозрительные комбинации
            suspicious_combos = ['\'', '"', '--', '/*', '*/', ';']
            if any(combo in prompt for combo in suspicious_combos):
                st.error("Обнаружена возможная SQL-инъекция")
                logging.warning(f"Возможная SQL-инъекция: {prompt[:100]}...")
                return False
            
        return True

    def validate_url(self, url: str) -> bool:
        if not url:
            return False
            
        try:
            if not url.startswith(('http://', 'https://')):
                return False
                
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
                
            return True
        except Exception:
            return False

    def validate_file_extension(self, filename: str) -> bool:
        allowed_extensions = {"txt", "pdf", "docx", "csv"}
        if PPTX_AVAILABLE:
            allowed_extensions.add("pptx")
        if XLSX_AVAILABLE:
            allowed_extensions.add("xlsx")
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ""
        return file_ext in allowed_extensions