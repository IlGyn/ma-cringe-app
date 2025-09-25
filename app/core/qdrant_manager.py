import uuid
import logging
from typing import List, Dict, Any
from collections import OrderedDict

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.message_manager import MessageManager
from app.core.encoder_manager import EncoderManager
from app.config import Config


class QdrantManager:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host="qdrant", port=6334, prefer_grpc=True)
        self.collection_name = collection_name

    # -------------------- Collection management -------------------- #
    def recreate_collection(self, vector_size: int) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            logging.info(f"Коллекция {self.collection_name} удалена")
        except Exception as e:
            logging.warning(f"Не удалось удалить коллекцию: {e}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )
        logging.info(f"Коллекция {self.collection_name} создана с размером вектора {vector_size}")

    def init_collection(self, encoder_manager: EncoderManager):
        """Инициализация коллекции с безопасной обработкой размерности"""
        # Используем безопасный метод, который всегда возвращает int
        vector_size = encoder_manager.get_sentence_embedding_dimension_safe(384)

        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.recreate_collection(vector_size)
                logging.info(f"Коллекция {self.collection_name} создана (не существовала)")
                return

            coll_info = self.client.get_collection(self.collection_name)
            coll_vectors = getattr(coll_info.config.params, 'vectors', None)

            recreate_needed: bool = False
            if isinstance(coll_vectors, qmodels.VectorParams):
                if coll_vectors.size != vector_size:
                    recreate_needed = True
                    logging.info(f"Размер вектора изменился: {coll_vectors.size} -> {vector_size}")
            elif isinstance(coll_vectors, dict):
                first_vec = next(iter(coll_vectors.values()), None)
                if first_vec and getattr(first_vec, 'size', None) != vector_size:
                    recreate_needed = True
                    logging.info(f"Размер вектора изменился: {getattr(first_vec, 'size', 'unknown')} -> {vector_size}")
            else:
                recreate_needed = True
                logging.info("Неизвестный формат векторов, пересоздаем коллекцию")

            if recreate_needed:
                logging.info(f"Пересоздаем коллекцию с новым размером вектора: {vector_size}")
                self.recreate_collection(vector_size)
            else:
                logging.info(f"Коллекция {self.collection_name} уже существует и корректна")

        except Exception as e:
            logging.error(f"Ошибка инициализации коллекции: {e}")
            self.recreate_collection(vector_size)

    # -------------------- Message handling -------------------- #
    def save_message(self, role: str, content: str, timestamp: str, encoder_manager: EncoderManager) -> None:
        if MessageManager.is_suspicious(content):
            logging.warning(f"Не сохраняем подозрительное сообщение: {content[:100]}...")
            return

        try:
            vector = encoder_manager.get_embedding(content)
            if not vector:
                logging.error("Не удалось получить эмбеддинг для сообщения")
                return
                
            point = qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"role": role, "content": content, "time": timestamp}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            logging.info(f"Сообщение сохранено: {role}, {timestamp}")
        except Exception as e:
            logging.error(f"Ошибка сохранения в Qdrant: {e}")

    def load_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Загрузка последних сообщений из коллекции"""
        try:
            hits = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [h.payload for h in hits[0] if h.payload is not None] if hits and hits[0] else []
        except Exception as e:
            logging.error(f"Ошибка загрузки сообщений: {e}")
            return []

    def clear_collection(self, encoder_manager: EncoderManager) -> None:
        """Очистка коллекции с безопасной обработкой ошибок"""
        try:
            # Пробуем удалить коллекцию
            try:
                self.client.delete_collection(self.collection_name)
                logging.info(f"Коллекция {self.collection_name} удалена")
            except Exception as e:
                logging.warning(f"Не удалось удалить коллекцию: {e}")

            # Используем безопасный метод для получения размерности
            vector_size = encoder_manager.get_sentence_embedding_dimension_safe(384)

            # Пересоздаем коллекцию
            self.recreate_collection(vector_size)
            logging.info(f"Коллекция {self.collection_name} очищена и пересоздана")
            
        except Exception as e:
            logging.error(f"Ошибка очистки коллекции: {e}")
            raise

    # -------------------- Context search -------------------- #
    def search_context(self, prompt: str, encoder_manager: EncoderManager, limit: int = 6) -> List[Dict[str, Any]]:
        cache_key = f"{prompt}_{limit}"
        if "context_cache" not in st.session_state:
            st.session_state.context_cache = OrderedDict()

        cached = st.session_state.context_cache.get(cache_key)
        if cached is not None:
            logging.info(f"Контекст для '{prompt}' загружен из кэша")
            return cached

        try:
            vector = encoder_manager.get_embedding(prompt)
            if not vector:
                logging.error("Не удалось получить эмбеддинг для поиска контекста")
                return []

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit * 2,
                with_payload=True
            )

            all_results = [p.payload for p in getattr(response, 'points', [])] if getattr(response, 'points', None) else []
            filtered = self._filter_and_cache(all_results, cache_key, limit)
            return filtered
        except Exception as e:
            logging.error(f"Ошибка поиска в Qdrant: {e}")
            return []

    def _filter_and_cache(self, results: List[Dict[str, Any]], cache_key: str, limit: int) -> List[Dict[str, Any]]:
        chat_messages = [msg for msg in results if msg.get('role') in ['user', 'assistant']]
        file_chunks = [msg for msg in results if msg.get('role') == 'file']

        recent_chat = chat_messages[-3:] if len(chat_messages) > 3 else chat_messages
        relevant_files = file_chunks[:3]

        combined = recent_chat + relevant_files
        filtered = [msg for msg in combined if not MessageManager.is_suspicious(msg.get('content', ''))]

        if len(filtered) != len(combined):
            logging.info(f"Отфильтровано {len(combined) - len(filtered)} сообщений из поиска")

        # Сохраняем в кэш
        st.session_state.context_cache[cache_key] = filtered
        if len(st.session_state.context_cache) > Config.MAX_CACHE_SIZE:
            st.session_state.context_cache.popitem(last=False)

        return filtered