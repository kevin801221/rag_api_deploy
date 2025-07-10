from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

# 導入模型和服務
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.request_models import DifyInferenceRequest
from models.response_models import DifyInferenceResponse, ErrorResponse
from services.inference_service import InferenceService

# 創建路由器
router = APIRouter(prefix="/api/v1/inference", tags=["inference"])

@router.post("/ask", response_model=DifyInferenceResponse, summary="智能問答 (Dify 格式)", description="使用 RAG 系統進行智能問答，符合 Dify 規範")
async def ask_question(request: DifyInferenceRequest):
    """
    RAG 推論 API (Dify 格式)
    
    接收 Dify 格式的用戶問題，基於知識庫返回答案
    
    - **knowledge_id**: 必填，知識庫 ID (對應 Qdrant collection name)
    - **query**: 必填，使用者提問內容
    - **retrieval_setting**: 可選，檢索設置，包含 top_k 和 score_threshold
    
    注意：需要 RAG 系統已初始化且知識庫中有數據
    """
    try:
        # 驗證請求
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查詢內容不能為空"
            )
        
        # 提取檢索設置
        retrieval_k = None
        score_threshold = None
        if request.retrieval_setting:
            retrieval_k = request.retrieval_setting.top_k
            score_threshold = request.retrieval_setting.score_threshold
        
        # 呼叫推論服務
        result = InferenceService.process_dify_query(
            knowledge_id=request.knowledge_id,
            query=request.query.strip(),
            retrieval_k=retrieval_k,
            score_threshold=score_threshold
        )
        
        if not result["success"]:
            # 根據錯誤類型返回適當的 HTTP 狀態碼
            if result.get("error_code") == "SYSTEM_NOT_READY":
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif result.get("error_code") == "KNOWLEDGE_ID_MISMATCH":
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            
            return JSONResponse(
                status_code=status_code,
                content=ErrorResponse(
                    error=result["error"],
                    error_code=result.get("error_code", "UNKNOWN_ERROR")
                ).dict()
            )
        
        # 返回成功結果 (Dify 格式)
        return DifyInferenceResponse(
            records=result["records"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"內部伺服器錯誤: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        # 檢查系統狀態
        is_ready = InferenceService.ensure_system_ready()
        
        if is_ready:
            from services.raptor_service import raptor_service
            stats = raptor_service.get_system_stats()
            
            return {
                "status": "healthy",
                "system_ready": True,
                "total_vectors": stats.get("vectorstore_stats", {}).get("total_vectors", 0),
                "message": "RAG 推論系統運行正常"
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "system_ready": False,
                    "message": "RAG 推論系統未就緒"
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "system_ready": False,
                "error": str(e)
            }
        )
