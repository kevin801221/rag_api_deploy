import os
import time
from typing import Dict, List, Optional, Tuple
from .raptor_service import raptor_service

# 導入核心模組
from raptor_core import ask_question, get_vectorstore_stats, _global_state
from models.response_models import Record # 導入新的 Record 模型

class InferenceService:
    """推論服務，處理問答業務邏輯"""
    
    @staticmethod
    def ensure_system_ready() -> bool:
        """確保系統就緒"""
        if not raptor_service.is_ready():
            print("🔄 系統未就緒，嘗試初始化...")
            return raptor_service.initialize_system()
        return True
    
    @staticmethod
    def process_dify_query(knowledge_id: str, query: str, retrieval_k: Optional[int] = None, score_threshold: Optional[float] = None) -> Dict:
        """處理 Dify 格式的查詢並返回結果"""
        start_time = time.time()
        
        try:
            # 確保系統就緒
            if not InferenceService.ensure_system_ready():
                return {
                    "success": False,
                    "error": "系統初始化失敗",
                    "error_code": "SYSTEM_NOT_READY"
                }
            
            # 這裡可以根據 knowledge_id 切換 Qdrant collection，如果有多個知識庫
            # 目前假設只有一個預設的 collection
            current_collection = os.getenv("QDRANT_COLLECTION", "rag_knowledge")
            if knowledge_id != current_collection:
                # 如果 knowledge_id 不匹配，可能需要重新設置 Qdrant 連接
                # 為了簡化，這裡先假設 knowledge_id 必須與配置的 collection_name 相同
                # 或者直接返回錯誤
                return {
                    "success": False,
                    "error": f"知識庫 ID 不匹配: {knowledge_id}. 當前知識庫為: {current_collection}",
                    "error_code": "KNOWLEDGE_ID_MISMATCH"
                }
            
            # 執行推論，獲取答案和相關文檔及分數
            # ask_question 現在返回 (answer, relevant_docs_with_scores)
            answer, relevant_docs_with_scores = ask_question(
                question=query,
                retrieval_k=retrieval_k,
                score_threshold=score_threshold
            )
            
            # 轉換為 Dify 的 Record 格式
            records = []
            for doc, score in relevant_docs_with_scores:
                # 提取元數據，確保是字典類型
                metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                
                records.append(Record(
                    content=doc.page_content,
                    score=score,
                    title=metadata.get('source', '未知來源'), # 使用 source 作為 title
                    metadata=metadata
                ).dict()) # 轉換為字典以便於返回
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "records": records,
                "processing_time": round(processing_time, 3)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ 推論失敗: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": "INFERENCE_FAILED",
                "processing_time": round(processing_time, 3)
            }