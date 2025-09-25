import logging
import numpy as np
import asyncio
from typing import List, Optional, Any, Union
import requests
import aiohttp
import streamlit as st
from app.core.embedding_cache import EmbeddingCache
from app.config import Config


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Any:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("Локальные эмбеддинги недоступны: установите torch и sentence-transformers") from e

    device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    logging.info(f"Загрузка модели эмбеддингов: {model_name} на {device}")
    return SentenceTransformer(model_name, device=device)


class EncoderManager:
    def __init__(self):
        self.cache = EmbeddingCache(max_size=Config.EMBEDDING_CACHE_SIZE)
        self._encoder: Optional[Any] = None
        self._provider: str = Config.EMBEDDINGS_PROVIDER.lower()
        self._ollama_base_url: str = Config.OLLAMA_URL
        self._ollama_model: str = Config.OLLAMA_EMBED_MODEL
        self._vector_size: Optional[int] = None
        self._model_dim_hints = {
            "all-minilm": 384,
            "nomic-embed-text": 768,
            "bge-m3": 1024,
            "mxbai-embed-large": 1024,
            "mxbai-embed-large-v1": 1024,
            "gte-small": 384,
            "gte-base": 768,
            "gte-large": 1024,
        }

    def reinitialize(self, provider: Optional[str] = None, ollama_model: Optional[str] = None):
        """Переинициализация менеджера с новыми настройками"""
        self._provider = provider if provider is not None else Config.EMBEDDINGS_PROVIDER.lower()
        self._ollama_model = ollama_model if ollama_model is not None else Config.OLLAMA_EMBED_MODEL
        self._encoder = None  # Сбрасываем кэшированный энкодер
        self._vector_size = None  # Сбрасываем размерность
        self.cache.clear()  # Очищаем кэш эмбеддингов

    # ---------- Encoder loading ----------
    def load_encoder(self, model_name: str = Config.DEFAULT_EMBEDDING_MODEL) -> Optional[Any]:
        if self._provider != "local":
            return None
        if self._encoder is None:
            self._encoder = load_sentence_transformer(model_name)
            try:
                self._vector_size = self._encoder.get_sentence_embedding_dimension()  # type: ignore
            except Exception:
                pass
        return self._encoder

    def get_encoder(self) -> Optional[Any]:
        return self.load_encoder(Config.DEFAULT_EMBEDDING_MODEL)

    # ---------- Embedding helpers ----------
    def _to_list(self, vec: Union[np.ndarray, List[float]]) -> List[float]:
        return np.array(vec).astype(float).tolist()

    def _get_or_compute_embedding_local(self, text: str, encoder: Any) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            logging.debug(f"Эмбеддинг из кэша: {text[:50]}...")
            return cached

        # Тут гарантированно encoder не None
        assert encoder is not None
        vec = encoder.encode(text, convert_to_numpy=False)  # type: ignore
        vec_list = self._to_list(vec)
        self.cache.put(text, vec_list)
        return vec_list

    def _get_or_compute_embedding_ollama(self, text: str) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        try:
            payload = {"model": self._ollama_model, "prompt": text}
            r = requests.post(f"{self._ollama_base_url}/api/embeddings", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            vector = data.get("embedding") or data.get("data", [{}])[0].get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Ollama embeddings: unexpected response format")
            vec_list = [float(x) for x in vector]
            self.cache.put(text, vec_list)
            if self._vector_size is None:
                self._vector_size = len(vec_list)
            return vec_list
        except Exception as e:
            logging.error(f"Ошибка получения эмбеддинга из Ollama: {e}")
            return []

    # ---------- Public API ----------
    def get_embedding(self, text: str, encoder: Optional[Any] = None) -> List[float]:
        if self._provider == "local":
            encoder = encoder or self.get_encoder()
            if encoder is None:
                raise RuntimeError("Локальный энкодер не инициализирован")
            return self._get_or_compute_embedding_local(text, encoder)
        return self._get_or_compute_embedding_ollama(text)

    async def async_encode(self, texts: List[str], encoder: Optional[Any] = None,
                           batch_size: int = 32) -> List[List[float]]:
        if self._provider == "ollama":
            return await self._async_encode_ollama(texts)

        encoder = encoder or self.get_encoder()
        assert encoder is not None
        cached_vectors: List[tuple[int, List[float]]] = []
        texts_to_encode: List[str] = []
        text_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                cached_vectors.append((i, cached))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)

        result_vectors: List[Union[List[float], None]] = [None] * len(texts)

        if texts_to_encode:
            vectors = await asyncio.to_thread(
                encoder.encode, texts_to_encode, batch_size=batch_size, convert_to_numpy=False  # type: ignore
            )
            processed_vectors = [self._to_list(vec) for vec in vectors]

            for idx, vec in cached_vectors:
                result_vectors[idx] = vec
            for i, vec in enumerate(processed_vectors):
                idx = text_indices[i]
                result_vectors[idx] = vec
                self.cache.put(texts_to_encode[i], vec)
        else:
            for idx, vec in cached_vectors:
                result_vectors[idx] = vec

        return [vec for vec in result_vectors if vec is not None]

    async def _async_encode_ollama(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        async with aiohttp.ClientSession() as session:
            async def fetch(idx: int, text: str):
                cached = self.cache.get(text)
                if cached is not None:
                    results[idx] = cached
                    return
                try:
                    payload = {"model": self._ollama_model, "prompt": text}
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with session.post(f"{self._ollama_base_url}/api/embeddings", json=payload, timeout=timeout) as resp:
                        data = await resp.json()
                        vector = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
                        if isinstance(vector, list):
                            vec_list = [float(x) for x in vector]
                            self.cache.put(text, vec_list)
                            if self._vector_size is None:
                                self._vector_size = len(vec_list)
                            results[idx] = vec_list
                        else:
                            results[idx] = []
                except Exception as e:
                    logging.error(f"Ollama embeddings error: {e}")
                    results[idx] = []

            await asyncio.gather(*[fetch(i, t) for i, t in enumerate(texts)])
        return [vec for vec in results if vec]

    def get_sentence_embedding_dimension(self) -> Optional[int]:
        """Возвращает размерность эмбеддингов или None если не может быть определена"""
        if self._provider == "local":
            enc = self.get_encoder()
            if enc is None:
                logging.error("Локальный энкодер не инициализирован")
                return None
            try:
                dimension = enc.get_sentence_embedding_dimension()  # type: ignore
                if dimension is not None:
                    self._vector_size = dimension
                return dimension
            except Exception as e:
                logging.error(f"Ошибка получения размерности локального энкодера: {e}")
                return None
        
        # Ollama
        if self._vector_size is not None:
            return self._vector_size
        
        model_key = (self._ollama_model or "").lower().split(":", 1)[0]
        hinted = self._model_dim_hints.get(model_key)
        if hinted is not None:
            self._vector_size = hinted
            return hinted

        # Последняя попытка: короткий текст
        try:
            test_embedding = self._get_or_compute_embedding_ollama("dimension_probe")
            if test_embedding and len(test_embedding) > 0:
                self._vector_size = len(test_embedding)
                return self._vector_size
            else:
                logging.warning("Не удалось получить тестовый эмбеддинг")
                return None
        except Exception as e:
            logging.warning(f"Не удалось определить размерность через пробу: {e}")
            return None

    def get_sentence_embedding_dimension_safe(self, default: int = 384) -> int:
        """Безопасная версия, которая всегда возвращает int"""
        dimension = self.get_sentence_embedding_dimension()
        if dimension is not None:
            return dimension
        logging.warning(f"Размерность неизвестна, используем значение по умолчанию: {default}")
        return default

    def validate_encoder(self) -> bool:
        """Проверяет, что энкодер работает корректно"""
        try:
            test_text = "test"
            embedding = self.get_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            logging.error(f"Ошибка валидации энкодера: {e}")
            return False