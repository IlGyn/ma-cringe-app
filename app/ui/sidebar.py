import streamlit as st
import json
from datetime import datetime
from app.config import Config

def render_sidebar(base_url, model, max_context, max_tokens, temperature, top_p, top_k, uploaded_files,
                  qdrant_manager, encoder_manager, file_processor, validator):
    with st.sidebar:
        st.header("⚙️ Настройки")
        base_url = st.text_input("Ollama URL", Config.OLLAMA_URL)
        model = st.text_input("Модель", Config.DEFAULT_LLM_MODEL)
        max_context = st.slider("Контекст для ответа", 2, 20, 6)
        embedder = st.text_input("Ollama embedding model", Config.OLLAMA_EMBED_MODEL)
        max_tokens = st.slider("Максимальная длина ответа", 50, 1000, 200)
        temperature = st.slider("Температура", 0.0, 1.5, 0.3)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.8)
        top_k = st.slider("Top-k", 1, 100, 40)

        if st.button("🔄 Перезагрузить энкодер"):
            try:
                # Явно передаем строки, а не None
                encoder_manager.reinitialize(
                    provider="ollama", 
                    ollama_model=str(embedder)  # Явное преобразование в строку
                )
                
                # Перезагружаем коллекцию Qdrant
                qdrant_manager.init_collection(encoder_manager)
                
                # Обновляем сессию
                st.session_state.encoder = encoder_manager.get_encoder()
                st.success(f"Эмбеддинги Ollama: {embedder}")
            except Exception as e:
                st.error(f"Ошибка перезагрузки энкодера: {e}")

        # Определяем поддерживаемые форматы динамически
        supported_formats = ["txt", "pdf", "docx", "csv"]
        try:
            from pptx import Presentation
            supported_formats.append("pptx")
        except ImportError:
            pass
        try:
            import openpyxl
            supported_formats.append("xlsx")
        except ImportError:
            pass
            
        uploaded_files = st.file_uploader(
            "📂 Загрузите файлы", 
            type=supported_formats, 
            accept_multiple_files=True
        )
        
        st.markdown("---")
        if st.button("💾 Экспорт чата"):
            if st.session_state.messages:
                chat_data = {
                    "exported_at": datetime.now().isoformat(),
                    "messages": st.session_state.messages,
                    "stats": st.session_state.stats
                }
                st.download_button(
                    label="📥 Скачать историю",
                    data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("Нет сообщений для экспорта")

        uploaded_chat = st.file_uploader("📤 Импорт чата", type=["json"])
        if uploaded_chat:
            try:
                chat_data = json.load(uploaded_chat)
                st.session_state.messages = chat_data.get("messages", [])
                if "stats" in chat_data:
                    st.session_state.stats = chat_data["stats"]
                st.success("История чата импортирована")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка импорта: {e}")
        
        if st.button("🗑 Очистить чат"):
            st.session_state.messages = []
            try:
                # УБЕДИТЕСЬ ЧТО ПЕРЕДАЕТСЯ encoder_manager, а не encoder
                qdrant_manager.clear_collection(encoder_manager)  # <-- Тут должно быть encoder_manager
                if "context_cache" in st.session_state:
                    st.session_state.context_cache.clear()
                if hasattr(encoder_manager, 'cache'):
                    encoder_manager.cache.clear()
                st.session_state.stats = {
                    "messages_sent": 0,
                    "files_processed": 0,
                    "chunks_saved": 0
                }
                st.success("История чата очищена")
                import logging
                logging.info("История чата очищена")
            except Exception as e:
                st.error(f"Ошибка при очистке Qdrant: {e}")
                import logging
                logging.error(f"Ошибка при очистке Qdrant: {e}")
            st.rerun()

        st.markdown("---")
        if st.button("📂 Показать загруженные файлы"):
            try:
                file_messages = []
                all_messages = qdrant_manager.load_messages(limit=100)
                for msg in all_messages:
                    if msg.get('role') == 'file':
                        file_messages.append(msg)
                
                if file_messages:
                    st.write(f"Загружено файлов: {len(set(msg.get('time', '') for msg in file_messages))}")
                    files_by_time = {}
                    for msg in file_messages:
                        time_key = msg.get('time', 'unknown')
                        if time_key not in files_by_time:
                            files_by_time[time_key] = []
                        files_by_time[time_key].append(msg)
                    
                    for time_key, chunks in files_by_time.items():
                        with st.expander(f"Файл от {time_key} ({len(chunks)} чанков)"):
                            if chunks:
                                st.text_area("Содержимое:", chunks[0].get('content', '')[:500] + "...", height=100, key=f"file_{time_key}")
                else:
                    st.info("Нет загруженных файлов")
            except Exception as e:
                st.error(f"Ошибка при получении файлов: {e}")

        st.markdown("---")
        st.subheader("📊 Статистика")
        st.write(f"Сообщений: {st.session_state.stats.get('messages_sent', 0)}")
        st.write(f"Файлов: {st.session_state.stats.get('files_processed', 0)}")
        st.write(f"Чанков: {st.session_state.stats.get('chunks_saved', 0)}")
        # Расширенная статистика кэша
        if hasattr(encoder_manager, 'cache'):
            cache_stats = encoder_manager.cache.get_stats()
            st.write(f"Кэш эмбеддингов: {cache_stats['size']}/{cache_stats['max_size']} ({cache_stats['usage_percent']:.1f}%)")

    return base_url, model, max_context, max_tokens, temperature, top_p, top_k, uploaded_files