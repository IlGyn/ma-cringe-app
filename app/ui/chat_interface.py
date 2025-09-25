import streamlit as st
from datetime import datetime
from queue import Queue
import threading
from app.config import Config
from app.clients.async_helpers import async_stream_thread
from app.core.AIState import AIState  # твой полноценный AIState

def render_chat_interface(prompt, qdrant_manager, encoder_manager, ollama_client, 
                          validator, file_processor, message_manager):

    # --- Инициализация AIState ---
    if "ai_state" not in st.session_state:
        st.session_state.ai_state = AIState(qdrant_manager, encoder_manager, user_id="default")

    # Статус Ollama
    if st.session_state.ollama_status:
        st.success(f"🟢 {st.session_state.ollama_message}")
    else:
        st.error(f"🔴 {st.session_state.ollama_message}")

    # --- Обработка файлов ---
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            result = file_processor.process_file(file, qdrant_manager, encoder_manager, Config.MAX_FILE_SIZE_MB)
            st.sidebar.success(result)

    # --- Отображение сообщений ---
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))
            st.caption(f"_{msg.get('time', '')}_")

    # --- Обработка ввода пользователя ---
    if st.session_state.ollama_status and (prompt := st.chat_input("Введите сообщение...")):
        if not validator.validate_prompt(prompt):
            st.rerun()

        ts = datetime.now().strftime("%H:%M:%S")

        # Сохраняем сообщение пользователя
        qdrant_manager.save_message("user", prompt, ts, encoder_manager)
        st.session_state.messages.append({"role": "user", "content": prompt, "time": ts})
        st.session_state.stats["messages_sent"] = st.session_state.stats.get("messages_sent", 0) + 1

        st.session_state.ai_state.add_message("user", prompt)

        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"_{ts}_")

        # --- Генерация ответа ---
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Подготовка ответа...")

            # Формируем контекст с памятью и личностью
            combined_prompt = st.session_state.ai_state.get_context_prompt(prompt)
            
            # Можно добавить дополнительный контекст из Qdrant
            context_messages = qdrant_manager.search_context(prompt, encoder_manager, st.session_state.max_context)
            if context_messages:
                combined_prompt += "\n\nДополнительный контекст:\n"
                for m in context_messages:
                    combined_prompt += f"{m['role']}: {m['content']}\n"

            progress_placeholder.empty()

            stop_button = st.empty()
            stop_generation = False
            q = Queue()
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=async_stream_thread,
                args=(q, cancel_event, ollama_client, combined_prompt, context_messages, st.session_state.model, 
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
                cancel_event.set()
                stop_button.empty()

            start_time = datetime.now()
            while not stop_generation:
                try:
                    chunk = q.get(timeout=0.5)
                    if chunk is None:
                        break
                    full_response += chunk
                    placeholder.markdown(full_response)
                    if (datetime.now() - start_time).seconds > Config.STREAM_TIMEOUT:
                        stop_generation = True
                        full_response += "\n\n⚠️ Превышено время ожидания"
                        break
                except:
                    if stop_generation or (datetime.now() - start_time).seconds > Config.STREAM_TIMEOUT:
                        break
                    continue

            try:
                thread.join(timeout=5)
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

        # Сохраняем ответ ассистента
        qdrant_manager.save_message("assistant", final_response, ts_reply, encoder_manager)
        st.session_state.messages.append({"role": "assistant", "content": final_response, "time": ts_reply})
        st.session_state.ai_state.add_message("assistant", final_response)

        st.rerun()