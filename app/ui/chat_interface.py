import streamlit as st
from datetime import datetime
from queue import Queue
import threading
from ..config import Config
from ..clients.async_helpers import async_stream_thread

def render_chat_interface(prompt, qdrant_manager, encoder_manager, ollama_client, 
                         validator, file_processor, message_manager):
    if st.session_state.ollama_status:
        st.success(f"🟢 {st.session_state.ollama_message}")
    else:
        st.error(f"🔴 {st.session_state.ollama_message}")

    # Обработка загрузки файлов
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            result = file_processor.process_file(file, qdrant_manager, encoder_manager, Config.MAX_FILE_SIZE_MB)
            st.sidebar.success(result)

    # Отображение сообщений
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))
            st.caption(f"_{msg.get('time', '')}_")

    # Обработка ввода пользователя
    if st.session_state.ollama_status and (prompt := st.chat_input("Введите сообщение...")):
        if not validator.validate_prompt(prompt):
            st.rerun()
        
        ts = datetime.now().strftime("%H:%M:%S")
        qdrant_manager.save_message("user", prompt, ts, encoder_manager)
        st.session_state.messages.append({"role": "user", "content": prompt, "time": ts})
        
        st.session_state.stats["messages_sent"] = st.session_state.stats.get("messages_sent", 0) + 1
        
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"_{ts}_")

        with st.chat_message("assistant"):
            # Индикатор прогресса для генерации
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Подготовка ответа...")
            
            with st.spinner("Генерация ответа..."):
                context = qdrant_manager.search_context(prompt, encoder_manager, st.session_state.max_context)
                progress_placeholder.empty()  # Убираем индикатор
                
                if Config.DEBUG_MODE:
                    with st.expander("🔍 Реальный контекст (отладка)", expanded=False):
                        st.write(f"Найдено сообщений: {len(context)}")
                        chat_count = sum(1 for msg in context if msg.get('role') in ['user', 'assistant'])
                        file_count = sum(1 for msg in context if msg.get('role') == 'file')
                        st.write(f"Чат: {chat_count}, Файлы: {file_count}")
                        
                        for i, msg in enumerate(context):
                            role = msg.get('role', 'unknown')
                            role_emoji = "👤" if role == "user" else "🤖" if role == "assistant" else "📄" if role == "file" else "❓"
                            st.write(f"**{i+1}. {role_emoji} {role.capitalize()}** ({msg.get('time', 'N/A')}):")
                            content = msg.get('content', '')
                            st.code(content[:300] + ('...' if len(content) > 300 else ''))

                stop_button = st.empty()
                stop_generation = False
                
                q = Queue()
                thread = threading.Thread(
                    target=async_stream_thread,
                    args=(q, ollama_client, prompt, context, st.session_state.model, 
                          st.session_state.base_url, st.session_state.max_tokens, 
                          st.session_state.temperature, st.session_state.top_p, st.session_state.top_k),
                    daemon=True
                )
                thread.start()

                full_response = ''
                placeholder = st.empty()
                
                stop_key = f"stop_{ts.replace(':', '_')}"
                if stop_button.button("⏹️ Остановить генерацию", key=stop_key):
                    stop_generation = True
                    stop_button.empty()
                
                # Таймер для таймаута
                start_time = datetime.now()
                
                while not stop_generation:
                    try:
                        chunk = q.get(timeout=0.5)  # Увеличенный таймаут
                        if chunk is None:
                            break
                        full_response += chunk
                        placeholder.markdown(full_response)
                        
                        # Проверка таймаута
                        if (datetime.now() - start_time).seconds > Config.STREAM_TIMEOUT:
                            stop_generation = True
                            full_response += "\n\n⚠️ Превышено время ожидания"
                            break
                            
                    except:
                        # Таймаут - проверяем нужно ли остановиться
                        if stop_generation or (datetime.now() - start_time).seconds > Config.STREAM_TIMEOUT:
                            break
                        continue
                
                # Улучшенное ожидание завершения потока
                try:
                    thread.join(timeout=5)  # Увеличенный таймаут
                except:
                    pass
                stop_button.empty()

        ts_reply = datetime.now().strftime("%H:%M:%S")
        if stop_generation and not full_response.strip():
            final_response = "Генерация была остановлена пользователем."
        elif not full_response.strip():
            final_response = "Извините, не удалось сгенерировать ответ."
        else:
            final_response = full_response.strip()

        qdrant_manager.save_message("assistant", final_response, ts_reply, encoder_manager)
        st.session_state.messages.append({"role": "assistant", "content": final_response, "time": ts_reply})
        st.rerun()