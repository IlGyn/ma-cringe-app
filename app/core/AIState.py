import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from app.core.qdrant_manager import QdrantManager
from app.core.encoder_manager import EncoderManager
from app.core.message_manager import MessageManager
import streamlit as st


class AIState:
    def __init__(self, qdrant_manager: QdrantManager, encoder_manager: EncoderManager, user_id: str = "default"):
        self.user_id = user_id
        self.qdrant_manager = qdrant_manager
        self.encoder_manager = encoder_manager

        self.user_facts: Dict[str, Any] = {}   # долговременные факты
        self.memory: List[Dict[str, str]] = []  # краткосрочные сообщения
        self.memory_limit = 200                 # максимум сообщений для краткосрочной памяти

        self.load_facts()

    # -------------------- Работа с памятью -------------------- #
    def add_message(self, role: str, content: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.memory.append({"role": role, "content": content, "time": ts})
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]

        if role == "user":
            self._analyze_and_store_fact(content)
            self.save_facts()

    def _analyze_and_store_fact(self, content: str):
        """
        Решаем, стоит ли запомнить сообщение как факт.
        Можно использовать ключевые слова, шаблоны или LLM.
        """
        content_lower = content.lower()
        
        # Простейший пример на ключевых словах
        possible_facts = {
            "любимая музыка": ["люблю", "мой любимый", "слушаю"],
            "хобби": ["хобби", "занимаюсь", "играю в"]
        }

        for fact_type, keywords in possible_facts.items():
            if any(word in content_lower for word in keywords):
                # Проверяем, не дублируем ли уже сохраненный факт
                if fact_type not in self.user_facts or self.user_facts[fact_type] != content:
                    self.user_facts[fact_type] = content
                    logging.info(f"Запомнен новый факт: {fact_type} -> {content}")

    def get_context_prompt(self, user_input: str) -> str:
        facts_text = "Вот что я знаю о пользователе:\n"
        for k, v in self.user_facts.items():
            facts_text += f"- {k}: {v}\n"

        memory_text = ""
        for m in self.memory[-6:]:  # последние 6 сообщений
            memory_text += f"{m['role']}: {m['content']}\n"

        personality_prompt = "Ты ИИ с характером: дружелюбный, полезный и ироничный.\n"
        return personality_prompt + facts_text + memory_text + f"User: {user_input}"

    # -------------------- Сохранение и загрузка -------------------- #
    def save_facts(self):
        try:
            with open(f"{self.user_id}_facts.json", "w", encoding="utf-8") as f:
                json.dump(self.user_facts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Ошибка сохранения фактов: {e}")

    def load_facts(self):
        try:
            with open(f"{self.user_id}_facts.json", "r", encoding="utf-8") as f:
                self.user_facts = json.load(f)
        except FileNotFoundError:
            self.user_facts = {}