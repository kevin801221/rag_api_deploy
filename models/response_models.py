# from pydantic import BaseModel, Field
# from typing import List, Optional
# from datetime import datetime

# class SourceInfo(BaseModel):
#     source: str = Field(description="來源文件路徑")
#     content_preview: str = Field(description="內容預覽")
#     relevance_score: Optional[float] = Field(None, description="相關性分數")

# class InferenceResponse(BaseModel):
#     success: bool = Field(description="是否成功")
#     answer: str = Field(description="RAG 生成的答案")
#     sources: Optional[List[SourceInfo]] = Field(None, description="參考來源")
#     processing_time: float = Field(description="處理時間(秒)")
#     total_vectors: Optional[int] = Field(None, description="知識庫向量總數")
#     timestamp: datetime = Field(default_factory=datetime.now)

# class ErrorResponse(BaseModel):
#     success: bool = False
#     error: str = Field(description="錯誤信息")
#     error_code: str = Field(description="錯誤代碼")
#     timestamp: datetime = Field(default_factory=datetime.now)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Dify 規範的 Record 格式
class Record(BaseModel):
    content: str = Field(description="文字片段")
    score: float = Field(description="相關性分數")
    title: str = Field(description="模索標題 (來源文件路徑)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="額外元數據")

# Dify 規範的推論響應格式
class DifyInferenceResponse(BaseModel):
    records: List[Record] = Field(description="檢索到的記錄列表")

# 原始 RAG 推論響應模型 (更名以避免衝突)
class RAGInferenceResponse(BaseModel):
    success: bool = Field(description="是否成功")
    answer: str = Field(description="RAG 生成的答案")
    sources: Optional[List[Record]] = Field(None, description="參考來源 (使用 Record 格式)")
    processing_time: float = Field(description="處理時間(秒)")
    total_vectors: Optional[int] = Field(None, description="知識庫向量總數")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ErrorResponse(BaseModel):
    success: bool = False
    error: str = Field(description="錯誤信息")
    error_code: str = Field(description="錯誤代碼")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# 更新相關響應模型
class ProcessedFileInfo(BaseModel):
    filename: str = Field(description="文件名")
    status: str = Field(description="處理狀態")
    chunks_created: Optional[int] = Field(None, description="創建的文本塊數量")
    processing_time: Optional[float] = Field(None, description="處理時間")
    error_message: Optional[str] = Field(None, description="錯誤信息")

class UpdateStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UpdateKnowledgeResponse(BaseModel):
    success: bool = Field(description="是否成功")
    task_id: Optional[str] = Field(None, description="任務ID（異步處理時）")
    message: str = Field(description="處理結果信息")
    processed_files: List[ProcessedFileInfo] = Field(default_factory=list, description="處理的文件信息")
    total_files_processed: int = Field(0, description="處理的文件總數")
    total_vectors_added: int = Field(0, description="新增的向量數量")
    total_vectors_in_db: int = Field(0, description="數據庫中向量總數")
    processing_time: float = Field(description="總處理時間")
    config_used: Optional[Dict[str, Any]] = Field(None, description="使用的配置")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class UpdateStatusResponse(BaseModel):
    task_id: str = Field(description="任務ID")
    status: UpdateStatus = Field(description="任務狀態")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="進度(0-1)")
    message: str = Field(description="狀態信息")
    result: Optional[UpdateKnowledgeResponse] = Field(None, description="完成後的結果")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
