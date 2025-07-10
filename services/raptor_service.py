import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 導入你現有的核心模組
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from raptor_core import (
    init_raptor_core, load_api_keys_from_files, setup_models, 
    setup_qdrant, build_rag_chain, get_vectorstore_stats,
    get_current_state, _global_state
)

class RaptorService:
    """RAPTOR 服務包裝類，處理系統初始化和狀態管理"""
    
    def __init__(self):
        self.initialized = False
        self.config = None
    
    def initialize_system(self, config: Dict = None) -> bool:
        """初始化 RAPTOR 系統"""
        try:
            print("🚀 初始化 RAG 推論系統...")
            
            # 使用默認配置或傳入的配置
            default_config = {
                "chunk_size": 1500,
                "chunk_overlap": 150,
                "n_levels": 3,
                "embedding_model": "text-embedding-3-small",
                "llm_model": "gpt-4o-mini",
                "retrieval_k": 6
            }
            
            self.config = config or default_config
            
            # 載入 API Keys
            api_keys = load_api_keys_from_files()
            if not api_keys:
                raise Exception("找不到 OpenAI API Key")
            
            # 初始化 RAPTOR 核心
            init_raptor_core(self.config)
            
            # 設置模型
            if not setup_models(openai_api_keys=api_keys):
                raise Exception("模型設置失敗")
            
            # 設置 Qdrant
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            collection_name = os.getenv("QDRANT_COLLECTION", "rag_knowledge")
            
            if not qdrant_url or not qdrant_api_key:
                raise Exception("缺少 Qdrant 配置")
            
            if not setup_qdrant(qdrant_url, qdrant_api_key, collection_name):
                raise Exception("Qdrant 連接失敗")
            
            # 建立 RAG 鏈
            if not build_rag_chain():
                raise Exception("RAG 鏈建立失敗")
            
            # 檢查知識庫
            stats = get_vectorstore_stats()
            if stats.get('total_vectors', 0) == 0:
                raise Exception("知識庫為空，請先更新知識庫")
            
            self.initialized = True
            print(f"✅ 系統初始化成功！知識庫有 {stats.get('total_vectors', 0):,} 個向量")
            return True
            
        except Exception as e:
            print(f"❌ 系統初始化失敗: {e}")
            self.initialized = False
            return False
    
    def is_ready(self) -> bool:
        """檢查系統是否就緒"""
        if not self.initialized:
            return False
        
        state = get_current_state()
        required_components = [
            'embd_loaded', 'model_loaded', 'qdrant_connected', 
            'vectorstore_ready', 'rag_chain_built'
        ]
        
        return all(state.get(component, False) for component in required_components)
    
    def get_system_stats(self) -> Dict:
        """獲取系統統計信息"""
        if not self.initialized:
            return {"error": "系統未初始化"}
        
        try:
            stats = get_vectorstore_stats()
            state = get_current_state()
            return {
                "vectorstore_stats": stats,
                "system_state": state,
                "config": self.config
            }
        except Exception as e:
            return {"error": f"獲取統計失敗: {e}"}

# 全局服務實例
raptor_service = RaptorService()