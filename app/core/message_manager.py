import logging
from typing import List, Dict, Any

class MessageManager:
    @staticmethod
    def load_messages(qdrant_manager, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            hits = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name, 
                limit=limit, 
                with_payload=True, 
                with_vectors=False
            )
            return [h.payload for h in hits[0] if h.payload is not None] if hits and hits[0] else []
        except Exception as e:
            logging.error(f"Ошибка загрузки сообщений: {e}")
            return []
    
    @staticmethod
    def truncate_context_by_tokens(context: List[Dict], max_tokens: int = 2000) -> List[Dict]:
        """Обрезает контекст по приблизительному количеству токенов"""
        total_chars = 0
        truncated = []
        
        for msg in reversed(context):
            content = msg.get('content', '')
            chars = len(content)
            if total_chars + chars > max_tokens * 4:
                break
            truncated.append(msg)
            total_chars += chars
        
        return list(reversed(truncated))

    @staticmethod
    def is_suspicious(content: str) -> bool:
        """Проверяет, содержит ли сообщение подозрительные фразы"""
        suspicious_phrases = [
            'согласно предыдущим взаимодействиям',
            'уже общаемся часто', 
            'знакомы с моими возможностями',
            'новый взгляд может быть очень ценным',
            'согласно последнему взаимодействию',
            'я только был загружен',
            'не имею воспоминаний'
        ]
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in suspicious_phrases)