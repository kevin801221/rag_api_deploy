# from pydantic import BaseModel, Field
# from typing import List, Optional
from enum import Enum
# from fastapi import UploadFile

# from models.config_models import RaptorConfig

class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"

# class UpdateKnowledgeRequest(BaseModel):
#     files: List[UploadFile]
#     config: Optional[RaptorConfig] = None
#     force_update: bool = False

# class InferenceRequest(BaseModel):
#     question: str = Field(..., min_length=1, max_length=1000, description="用戶問題")
#     retrieval_k: Optional[int] = Field(6, ge=1, le=20, description="檢索文檔數量")
#     show_sources: Optional[bool] = Field(True, description="是否返回來源信息")
#     temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, description="生成溫度")

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# 原始 RAG 推論請求模型 (更名以避免衝突)
class RAGInferenceRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用戶問題")
    retrieval_k: Optional[int] = Field(6, ge=1, le=20, description="檢索文檔數量")
    show_sources: Optional[bool] = Field(True, description="是否返回來源信息")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, description="生成溫度")

# Dify 規範的檢索設置
class RetrievalSetting(BaseModel):
    top_k: int = Field(5, ge=1, le=20, description="檢索結果數量")
    score_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="檢索分數閾值")

# Dify 規範的推論請求模型
class DifyInferenceRequest(BaseModel):
    knowledge_id: str = Field(..., description="知識庫 ID (對應 Qdrant collection name)")
    query: str = Field(..., min_length=1, description="使用者提問內容")
    retrieval_setting: Optional[RetrievalSetting] = Field(None, description="檢索設置")

# 知識庫更新請求模型
class UpdateConfigRequest(BaseModel):
    chunk_size: Optional[int] = Field(1500, ge=100, le=5000, description="文本分塊大小")
    chunk_overlap: Optional[int] = Field(150, ge=0, le=1000, description="文本重疊大小")
    n_levels: Optional[int] = Field(3, ge=1, le=5, description="RAPTOR 層數")
    embedding_model: Optional[str] = Field("text-embedding-3-small", description="嵌入模型")
    llm_model: Optional[str] = Field("gpt-4o-mini", description="語言模型")
    force_update: Optional[bool] = Field(False, description="是否強制更新所有文件")

class UpdateKnowledgeRequest(BaseModel):
    config: Optional[UpdateConfigRequest] = Field(None, description="更新配置")
    target_files: Optional[List[str]] = Field(None, description="指定要更新的文件名列表")
