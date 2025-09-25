import streamlit as st
from datetime import datetime
from collections import OrderedDict

# Импорт компонентов
from app.config import Config
from app.logging_config import setup_logging
from app.core.message_manager import MessageManager
from app.core.qdrant_manager import QdrantManager
from app.core.encoder_manager import EncoderManager
from app.clients.ollama_client import OllamaClient
from app.processors.file_processor import FileProcessor
from app.processors.validators import Validator
from app.ui.sidebar import render_sidebar
from app.ui.chat_interface import render_chat_interface
from app.core.AIState import AIState

# Настройка логирования
setup_logging()

# Инициализация компонентов
config = Config()
# В Docker-сети используем имя контейнера вместо localhost
qdrant_host = config.QDRANT_HOST
qdrant_manager = QdrantManager(qdrant_host, config.QDRANT_PORT, config.COLLECTION_NAME)
encoder_manager = EncoderManager()
ollama_client = OllamaClient()
file_processor = FileProcessor()
validator = Validator()
message_manager = MessageManager()

def main():
    st.set_page_config(page_title="LLM Chat", layout="wide")
    st.title("💬 LLM Чат")

    if config.DEBUG_MODE:
        st.sidebar.info("DEBUG режим включен")

    # Инициализация состояния сессии
    if "encoder" not in st.session_state:
        # Для provider='ollama' вернётся None, это ок
        st.session_state.encoder = encoder_manager.load_encoder(config.DEFAULT_EMBEDDING_MODEL)
        qdrant_manager.init_collection(encoder_manager)

    if "messages" not in st.session_state:
        try:
            st.session_state.messages = message_manager.load_messages(qdrant_manager)
        except Exception as e:
            import logging
            logging.warning(f"Не удалось загрузить сообщения: {e}")
            st.session_state.messages = []

    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status, st.session_state.ollama_message = ollama_client.check_api(config.OLLAMA_URL)

    if "context_cache" not in st.session_state:
        st.session_state.context_cache = OrderedDict()
        st.session_state.context_cache_max_size = config.MAX_CACHE_SIZE

    if "stats" not in st.session_state:
        st.session_state.stats = {
            "messages_sent": 0,
            "files_processed": 0,
            "chunks_saved": 0
        }

    # Инициализация настроек по умолчанию
    if "base_url" not in st.session_state:
        st.session_state.base_url = config.OLLAMA_URL
    if "model" not in st.session_state:
        st.session_state.model = config.DEFAULT_LLM_MODEL
    if "max_context" not in st.session_state:
        st.session_state.max_context = 6
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 200
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.8
    if "top_k" not in st.session_state:
        st.session_state.top_k = 40

    if "ai_state" not in st.session_state:
        st.session_state.ai_state = AIState(qdrant_manager, encoder_manager)

    # Рендеринг интерфейса
    (st.session_state.base_url, st.session_state.model, st.session_state.max_context,
    st.session_state.max_tokens, st.session_state.temperature, st.session_state.top_p,
    st.session_state.top_k, st.session_state.uploaded_files) = render_sidebar(
        st.session_state.base_url, st.session_state.model, st.session_state.max_context,
        st.session_state.max_tokens, st.session_state.temperature, st.session_state.top_p,
        st.session_state.top_k, st.session_state.uploaded_files if "uploaded_files" in st.session_state else None,
        qdrant_manager, encoder_manager, file_processor, validator
    )

    # Рендеринг чат-интерфейса
    render_chat_interface(
        None, qdrant_manager, encoder_manager, ollama_client, 
        validator, file_processor, message_manager
    )

if __name__ == "__main__":
    main()