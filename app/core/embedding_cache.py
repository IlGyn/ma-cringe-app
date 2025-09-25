from collections import OrderedDict

class EmbeddingCache:
    def __init__(self, max_size: int = 500):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, text: str):
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        return None

    def put(self, text: str, vector):
        self.cache[text] = vector
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)
        
    def get_stats(self) -> dict:
        """Возвращает статистику кэша"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0
        }