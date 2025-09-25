import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from app.core.embedding_cache import EmbeddingCache

class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self.cache = EmbeddingCache(max_size=3)

    def test_put_and_get(self):
        self.cache.put("test1", [1.0, 2.0, 3.0])
        result = self.cache.get("test1")
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_cache_eviction(self):
        self.cache.put("test1", [1.0])
        self.cache.put("test2", [2.0])
        self.cache.put("test3", [3.0])
        self.cache.put("test4", [4.0])  # Should evict test1
        
        self.assertIsNone(self.cache.get("test1"))
        self.assertIsNotNone(self.cache.get("test2"))

    def test_move_to_end(self):
        self.cache.put("test1", [1.0])
        self.cache.put("test2", [2.0])
        self.cache.get("test1")  # Move to end
        self.cache.put("test3", [3.0])
        self.cache.put("test4", [4.0])  # Should evict test2, not test1
        
        self.assertIsNotNone(self.cache.get("test1"))
        self.assertIsNone(self.cache.get("test2"))

    def test_clear(self):
        self.cache.put("test1", [1.0])
        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)

    def test_get_stats(self):
        stats = self.cache.get_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["usage_percent"], 0.0)

if __name__ == '__main__':
    unittest.main()