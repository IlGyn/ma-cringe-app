import uuid
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from .message_manager import MessageManager
from collections import OrderedDict
import streamlit as st
from ..config import Config

class QdrantManager:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port, prefer_grpc=True)
        self.collection_name = collection_name

    def recreate_collection(self, vector_size: int):
        try:
            self.client.delete_collection(self.collection_name)
            logging.info(f"Коллекция {self.collection_name} удалена")
        except Exception as e:
            logging.warning(f"Не удалось удалить коллекцию: {e}")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )
        logging.info(f"Коллекция {self.collection_name} создана")

    def init_collection(self, encoder):
        vector_size = encoder.get_sentence_embedding_dimension()
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.recreate_collection(vector_size)
                return

            coll_info = self.client.get_collection(self.collection_name)
            coll_vectors = getattr(coll_info.config.params, 'vectors', None)

            recreate_needed = False
            if isinstance(coll_vectors, qmodels.VectorParams):
                if coll_vectors.size != vector_size:
                    recreate_needed = True
            elif isinstance(coll_vectors, dict):
                first_vec = next(iter(coll_vectors.values()), None)
                if first_vec and getattr(first_vec, 'size', None) != vector_size:
                    recreate_needed = True
            else:
                recreate_needed = True

            if recreate_needed:
                self.recreate_collection(vector_size)

        except Exception as e:
            logging.error(f"Ошибка инициализации коллекции: {e}")
            self.recreate_collection(vector_size)

    def save_message(self, role: str, content: str, timestamp: str, encoder_manager):
        # ФИЛЬТР: не сохраняем странные ответы
        if MessageManager.is_suspicious(content):
            logging.warning(f"Не сохраняем подозрительное сообщение: {content[:100]}...")
            return
        
        try:
            vector = encoder_manager.get_embedding(content)
            point = qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"role": role, "content": content, "time": timestamp}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            logging.info(f"Сообщение сохранено: {role}, {timestamp}")
        except Exception as e:
            logging.error(f"Ошибка сохранения в Qdrant: {e}")

    def search_context(self, prompt: str, encoder_manager, limit: int = 6) -> List[Dict[str, Any]]:
        # Проверка кэша
        cache_key = f"{prompt}_{limit}"
        if "context_cache" not in st.session_state:
            st.session_state.context_cache = OrderedDict()
        
        cached = st.session_state.context_cache.get(cache_key)
        if cached is not None:
            logging.info(f"Контекст для '{prompt}' загружен из кэша")
            return cached
        
        try:
            vector = encoder_manager.get_embedding(prompt)
  
            response = self.client.query_points(
                collection_name=self.collection_name, 
                query=vector, 
                limit=limit * 2,
                with_payload=True
            )
            
            all_results = [p.payload for p in getattr(response, 'points', [])] if getattr(response, 'points', None) is not None else []
            
            chat_messages = [msg for msg in all_results if msg.get('role') in ['user', 'assistant']]
            file_chunks = [msg for msg in all_results if msg.get('role') == 'file']
            
            recent_chat = chat_messages[-3:] if len(chat_messages) > 3 else chat_messages
            relevant_files = file_chunks[:3]
            
            result = recent_chat + relevant_files
            
            # ФИЛЬТРАЦИЯ: убираем подозрительные сообщения
            filtered_result = [msg for msg in result if not MessageManager.is_suspicious(msg.get('content', ''))]
            
            if len(filtered_result) != len(result):
                logging.info(f"Отфильтровано {len(result) - len(filtered_result)} сообщений из поиска")
            
            # Сохраняем в кэш
            st.session_state.context_cache[cache_key] = filtered_result
            if len(st.session_state.context_cache) > Config.MAX_CACHE_SIZE:
                st.session_state.context_cache.popitem(last=False)
            
            return filtered_result
        except Exception as e:
            logging.error(f"Ошибка поиска в Qdrant: {e}")
            return []

    def load_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Загрузка всех сообщений из коллекции"""
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

    def clear_collection(self, encoder):
        try:
            self.client.delete_collection(self.collection_name)
            self.recreate_collection(encoder.get_sentence_embedding_dimension())
        except Exception as e:
            logging.error(f"Ошибка очистки коллекции: {e}")