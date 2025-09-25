import logging
from app.config import Config

def setup_logging():
    """Настройка логирования для приложения"""
    logging.basicConfig(
        filename="chat_app.log",
        level=logging.INFO if not Config.DEBUG_MODE else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
