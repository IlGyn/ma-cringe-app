import logging
import numpy as np
import asyncio
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st
from .embedding_cache import EmbeddingCache
from ..config import Config

@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Загрузка модели эмбеддингов: {model_name} на {device}")
    return SentenceTransformer(model_name, device=device)

class EncoderManager:
    def __init__(self):
        self.cache = EmbeddingCache(max_size=Config.EMBEDDING_CACHE_SIZE)
        self._encoder = None

    def load_encoder(self, model_name: str = "all-MiniLM-L6-v2"):
        return load_sentence_transformer(model_name)

    def get_encoder(self):
        if self._encoder is None:
            self._encoder = self.load_encoder(Config.DEFAULT_EMBEDDING_MODEL)
        return self._encoder

    def get_embedding(self, text: str, encoder=None) -> List[float]:
        if encoder is None:
            encoder = self.get_encoder()
            
        # Проверка кэша
        cached = self.cache.get(text)
        if cached is not None:
            logging.debug(f"Эмбеддинг из кэша: {text[:50]}...")
            return cached
        
        # Генерация эмбеддинга
        vec = encoder.encode(text, convert_to_numpy=False)
        vec_list: List[float] = []
        
        if hasattr(vec, 'tolist'):
            vec_list = [float(x) for x in vec.tolist()]
        elif isinstance(vec, (list, np.ndarray)):
            vec_list = [float(x) for x in vec]
        else:
            vec_list = [float(vec)]

        # Сохраняем в кэш
        self.cache.put(text, vec_list)
        return vec_list

    async def async_encode(self, texts: List[str], encoder=None, batch_size: int = 32) -> List[List[float]]:
        """Асинхронная генерация эмбеддингов"""
        if encoder is None:
            encoder = self.get_encoder()
            
        loop = asyncio.get_event_loop()
        
        # Проверяем кэш для каждого текста
        cached_vectors: List[tuple] = []
        texts_to_encode: List[str] = []
        text_indices: List[int] = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                cached_vectors.append((i, cached))
                logging.debug(f"Эмбеддинг из кэша: {text[:50]}...")
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        result_vectors: List[Union[List[float], None]] = [None] * len(texts)
        
        if texts_to_encode:
            vectors = await loop.run_in_executor(
                None, 
                lambda: encoder.encode(texts_to_encode, batch_size=batch_size, convert_to_numpy=False)
            )
            
            if hasattr(vectors, 'tolist'):
                vectors = vectors.tolist()
            elif not isinstance(vectors, list):
                vectors = list(vectors)
            
            processed_vectors: List[List[float]] = []
            for vec in vectors:
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                if isinstance(vec, (list, np.ndarray)):
                    processed_vectors.append([float(x) for x in vec])
                else:
                    processed_vectors.append([float(vec)])
            
            for idx, vec in cached_vectors:
                result_vectors[idx] = vec
            
            for i, vec in enumerate(processed_vectors):
                text_idx = text_indices[i]
                result_vectors[text_idx] = vec
                self.cache.put(texts_to_encode[i], vec)
        else:
            for idx, vec in cached_vectors:
                result_vectors[idx] = vec
        
        final_result: List[List[float]] = []
        for vec in result_vectors:
            if vec is not None:
                final_result.append(vec)
        
        return final_result