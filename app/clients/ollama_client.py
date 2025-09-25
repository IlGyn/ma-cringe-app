import requests
import socket
import aiohttp
import asyncio
import json
import logging
from typing import AsyncGenerator, Tuple, Optional, Any
from urllib.parse import urlparse
from app.core.message_manager import MessageManager
from app.config import Config
from app.core.AIState import AIState

class OllamaClient:
    def check_connection(self, base_url: str, timeout: int = 2) -> bool:
        """Проверяет доступность Ollama на основе OLLAMA_URL.

        Поддерживает значения вроде http://localhost:11434, http://host.docker.internal:11434 и т.п.
        """
        try:
            parsed = urlparse(base_url)
            host: str = parsed.hostname or "127.0.0.1"
            # Если порт не указан, используем дефолтный порт Ollama
            port: int = parsed.port or 11434

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                return sock.connect_ex((host, port)) == 0
        except Exception as e:
            logging.error(f"Ошибка проверки порта: {e}")
            return False

    def check_api(self, base_url: str) -> Tuple[bool, str]:
        if not self.check_connection(base_url):
            return False, "Соединение с Ollama недоступно"
        try:
            r = requests.get(f"{base_url}/api/version", timeout=2)
            if r.status_code == 200:
                return True, "Подключение установлено"
            else:
                return False, "Нет ответа от сервера"
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def get_context_prompt(self, user_input: str, ai_state: AIState) -> str:
        return ai_state.get_context_prompt(user_input)

    async def ask_llm_async_stream(self, prompt: str, context: list, model: str, 
                                    base_url: str, max_tokens: int = 200,
                                    temperature: float = 0.3, top_p: float = 0.8, 
                                    top_k: int = 40, cancel_event: Optional[Any] = None) -> AsyncGenerator[str, None]:

        system_instruction = (
            "Ты — виртуальный помощник. Отвечай по существу, используя предоставленный контекст из файлов и истории чата.\n"
            "Если в контексте есть релевантная информация из загруженных файлов, используй её (например, 'Согласно загруженному файлу...').\n"
            "Если контекста нет или он не релевантен, отвечай на основе общего знания.\n"
            "НЕ добавляй фразы типа 'Согласно предыдущим взаимодействиям' или 'Как дела у вас?'.\n"
            "Отвечай напрямую на вопрос пользователя."
        )
        
        # Фильтрация контекста
        filtered_context = [msg for msg in context if not MessageManager.is_suspicious(msg.get('content', ''))]
        
        if len(filtered_context) != len(context):
            logging.info(f"Отфильтровано {len(context) - len(filtered_context)} сообщений для prompt")
            context = filtered_context
        
        full_prompt = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" 
            for m in context 
            if m.get('content') and m.get('content').strip()
        ] + [f"User: {prompt}", "Assistant:"])
        
        data = {
            "model": model,
            "prompt": f"System: {system_instruction}\n{full_prompt}",
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": 8192
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                timeout = aiohttp.ClientTimeout(total=Config.STREAM_TIMEOUT, connect=30)
                async with session.post(f"{base_url}/api/generate", json=data, timeout=timeout) as r:
                    if r.status != 200:
                        error_text = await r.text()
                        yield f"⚠️ Ошибка Ollama ({r.status}): {error_text}"
                        return
                        
                    async for raw_line in r.content:
                        # Поддержка отмены
                        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                            yield "⏹️ Генерация остановлена пользователем"
                            break
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data:'):
                                json_str = line_str[5:].strip()
                            else:
                                json_str = line_str
                                
                            if json_str:
                                chunk = json.loads(json_str)
                                if 'error' in chunk:
                                    yield f"⚠️ Ошибка модели: {chunk['error']}"
                                    break
                                elif chunk.get("response"):
                                    yield chunk["response"]
                                elif chunk.get("done"):
                                    break
                        except json.JSONDecodeError as je:
                            logging.warning(f"Не JSON данные: {line_str[:100]}...")
                            continue
                        except Exception as e:
                            logging.error(f"Ошибка обработки chunk: {e}")
                            yield f"⚠️ Ошибка обработки: {e}"
                            break
            except asyncio.TimeoutError:
                yield "⚠️ Таймаут при запросе к модели"
            except Exception as e:
                error_msg = f"⚠️ Ошибка Ollama: {str(e)}"
                logging.error(f"Ошибка Ollama: {e}")
                yield error_msg