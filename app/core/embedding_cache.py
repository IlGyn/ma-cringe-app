from collections import OrderedDict
from typing import List, Optional, Any, Dict

class EmbeddingCache:
    def __init__(self, max_size: int = 500):
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.max_size: int = max_size

    def get(self, text: str) -> Optional[List[float]]:
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        return None

    def put(self, text: str, vector: List[float]) -> None:
        self.cache[text] = vector
        self.cache.move_to_end(text)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def remove(self, text: str) -> None:
        """Удаляет конкретный элемент из кэша, если он есть"""
        self.cache.pop(text, None)

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        usage_percent = (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": usage_percent
        }
