import asyncio
import threading
from queue import Queue

def async_stream_thread(q: Queue, *args):
    async def inner_gen():
        try:
            async for token in args[0].ask_llm_async_stream(*args[1:]):
                q.put(token)
        except Exception as e:
            import logging
            logging.error(f"Ошибка в inner_gen: {e}")
            q.put(f"⚠️ Ошибка стрима: {e}")
        finally:
            q.put(None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(inner_gen())
    finally:
        loop.close()