# ================================
# routes/update.py (修復文件上傳問題)
# ================================
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
import json

from models.request_models import UpdateKnowledgeRequest, UpdateConfigRequest
from models.response_models import (
    UpdateKnowledgeResponse, UpdateStatusResponse, ErrorResponse, 
    ProcessedFileInfo, UpdateStatus
)
from services.update_service import UpdateService

# 創建路由器
router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

@router.post("/update", response_model=UpdateKnowledgeResponse, summary="更新知識庫", description="上傳和處理文件到知識庫，支援大文件和异步處理")
async def update_knowledge(
    background_tasks: BackgroundTasks,
    files: Union[List[UploadFile], None] = File(default=None),
    config: Optional[str] = Form(default=None),
    target_files: Optional[str] = Form(default=None),
    async_processing: Optional[bool] = Form(default=False)
):
    """
    更新知識庫（支援文件上傳）
    
    - **files**: 上傳的文件（PDF, TXT, DOCX）
    - **config**: 可選，自定義配置（JSON 格式）
    - **target_files**: 可選，要處理的目標文件列表（JSON 格式）
    - **async_processing**: 可選，是否异步處理，適用於大文件
    
    注意：此端點會修改知識庫內容，請謹慎使用
    - **config**: JSON格式的配置參數，例如: {"chunk_size": 1000, "force_update": true}
    - **target_files**: JSON格式的目標文件名列表，例如: ["file1.pdf", "file2.txt"]
    - **async_processing**: 是否異步處理（大文件建議使用）
    
    使用範例:
    ```bash
    # 空請求（檢查現有文件）
    curl -X POST "http://localhost:8000/api/v1/knowledge/update"
    
    # 上傳文件
    curl -X POST "http://localhost:8000/api/v1/knowledge/update" -F "files=@document.pdf"
    
    # 強制更新
    curl -X POST "http://localhost:8000/api/v1/knowledge/update" -F 'config={"force_update": true}'
    
    # 處理特定文件
    curl -X POST "http://localhost:8000/api/v1/knowledge/update" -F 'target_files=["doc1.pdf"]'
    ```
    """
    try:
        print(f"📝 收到更新請求:")
        print(f"   files: {files}")
        print(f"   files count: {len(files) if files else 0}")
        print(f"   config: {config}")
        print(f"   target_files: {target_files}")
        print(f"   async_processing: {async_processing}")
        
        # 處理文件參數 - 修復邏輯
        uploaded_files = []
        if files:
            # 過濾掉空文件或無檔名的文件
            valid_files = [f for f in files if f.filename and f.filename.strip()]
            
            if valid_files:
                try:
                    uploaded_files = await UpdateService.save_uploaded_files(valid_files)
                    print(f"✅ 成功上傳 {len(uploaded_files)} 個文件: {uploaded_files}")
                except ValueError as e:
                    print(f"❌ 文件類型錯誤: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(e)
                    )
                except Exception as e:
                    print(f"❌ 文件上傳失敗: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"文件上傳失敗: {str(e)}"
                    )
            else:
                print("   沒有有效的文件需要上傳")
        
        # 解析配置參數
        config_dict = None
        if config and config.strip():
            try:
                config_data = json.loads(config)
                # 只允許特定的配置參數
                allowed_config_keys = [
                    'chunk_size', 'chunk_overlap', 'n_levels', 
                    'embedding_model', 'llm_model', 'force_update'
                ]
                config_dict = {
                    k: v for k, v in config_data.items() 
                    if k in allowed_config_keys
                }
                print(f"   解析配置: {config_dict}")
            except json.JSONDecodeError as e:
                print(f"❌ 配置解析錯誤: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"配置參數格式錯誤，請使用有效的 JSON: {str(e)}"
                )
        
        # 解析目標文件列表
        target_file_list = None
        if target_files and target_files.strip():
            try:
                target_file_list = json.loads(target_files)
                if not isinstance(target_file_list, list):
                    raise ValueError("target_files 必須是字符串列表")
                print(f"   目標文件: {target_file_list}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ 目標文件解析錯誤: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"target_files 格式錯誤，請使用 JSON 字符串列表: {str(e)}"
                )
        
        # 如果有上傳文件且沒有指定 target_files，則處理上傳的文件
        if uploaded_files and not target_file_list:
            target_file_list = uploaded_files
            print(f"   將處理上傳的文件: {target_file_list}")
        
        # 如果沒有任何輸入，執行默認更新邏輯
        if not uploaded_files and not target_file_list and not config_dict:
            print("   執行默認更新邏輯（檢查現有文件）")
        
        # 異步處理
        if async_processing:
            task_id = UpdateService.create_task_id()
            UpdateService.update_task_status(task_id, "pending", 0.0, "任務已創建")
            
            # 添加背景任務
            background_tasks.add_task(
                UpdateService.process_knowledge_update_async,
                task_id,
                config_dict,
                target_file_list
            )
            
            return UpdateKnowledgeResponse(
                success=True,
                task_id=task_id,
                message="任務已創建，正在異步處理中",
                processing_time=0.0
            )
        
        # 同步處理
        print("🔄 開始同步處理...")
        result = UpdateService.process_knowledge_update_sync(
            config=config_dict,
            target_files=target_file_list
        )
        
        # 構建響應
        processed_files = []
        if uploaded_files:
            for filename in uploaded_files:
                processed_files.append(ProcessedFileInfo(
                    filename=filename,
                    status="processed" if result["success"] else "failed",
                    error_message=None if result["success"] else result["message"]
                ))
        
        response = UpdateKnowledgeResponse(
            success=result["success"],
            message=result["message"],
            processed_files=processed_files,
            total_files_processed=result["total_files_processed"],
            total_vectors_in_db=result["total_vectors_in_db"],
            processing_time=result["processing_time"],
            config_used=result.get("config_used")
        )
        
        print(f"✅ 響應: {response.message}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 內部錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"內部伺服器錯誤: {str(e)}"
        )

@router.post("/update-simple", response_model=UpdateKnowledgeResponse, summary="簡化知識庫更新", description="簡化的知識庫更新端點，僅支援 JSON 請求，不支援文件上傳")
async def update_knowledge_simple(request: UpdateKnowledgeRequest):
    """
    簡化的知識庫更新端點（僅 JSON，不支援文件上傳）
    
    適用於：
    - 更新現有文件的配置
    - 強制重新處理現有文件
    - 處理指定的文件列表
    
    不適用於：
    - 上傳新文件（請使用 /update 端點）
    
    使用範例:
    ```bash
    # 空請求（檢查現有文件）
    curl -X POST "http://localhost:8000/api/v1/knowledge/update-simple" \\
         -H "Content-Type: application/json" -d '{}'
    
    # 強制更新所有文件
    curl -X POST "http://localhost:8000/api/v1/knowledge/update-simple" \\
         -H "Content-Type: application/json" \\
         -d '{"config": {"force_update": true}}'
    
    # 修改配置並重新處理
    curl -X POST "http://localhost:8000/api/v1/knowledge/update-simple" \\
         -H "Content-Type: application/json" \\
         -d '{"config": {"chunk_size": 1000, "n_levels": 2}}'
    
    # 處理特定文件
    curl -X POST "http://localhost:8000/api/v1/knowledge/update-simple" \\
         -H "Content-Type: application/json" \\
         -d '{"target_files": ["document1.pdf", "document2.txt"]}'
    ```
    """
    try:
        print(f"📝 收到簡化更新請求: {request}")
        
        # 轉換配置
        config_dict = None
        if request.config:
            config_dict = request.config.dict(exclude_none=True)
        
        # 同步處理
        result = UpdateService.process_knowledge_update_sync(
            config=config_dict,
            target_files=request.target_files
        )
        
        return UpdateKnowledgeResponse(
            success=result["success"],
            message=result["message"],
            total_files_processed=result["total_files_processed"],
            total_vectors_in_db=result["total_vectors_in_db"],
            processing_time=result["processing_time"],
            config_used=result.get("config_used")
        )
        
    except Exception as e:
        print(f"❌ 簡化更新失敗: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新失敗: {str(e)}"
        )

@router.get("/status/{task_id}", response_model=UpdateStatusResponse, summary="查詢任務狀態", description="查詢异步知識庫更新任務的狀態和進度")
async def get_update_status(task_id: str):
    """
    獲取异步更新任務狀態
    
    - **task_id**: 任務 ID，由 /update 端點在异步模式下返回
    
    返回任務的狀態、進度和結果（如果已完成）
    """
    try:
        task_info = UpdateService.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任務不存在"
            )
        
        # 轉換結果
        result = None
        if task_info.get("result"):
            result_data = task_info["result"]
            result = UpdateKnowledgeResponse(
                success=result_data["success"],
                message=result_data["message"],
                total_files_processed=result_data["total_files_processed"],
                total_vectors_in_db=result_data["total_vectors_in_db"],
                processing_time=result_data["processing_time"],
                config_used=result_data.get("config_used")
            )
        
        return UpdateStatusResponse(
            task_id=task_id,
            status=UpdateStatus(task_info["status"]),
            progress=task_info.get("progress"),
            message=task_info["message"],
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取任務狀態失敗: {str(e)}"
        )

@router.get("/info", summary="知識庫信息", description="獲取知識庫的統計信息和狀態")
async def get_knowledge_info():
    """
    獲取知識庫信息
    
    返回知識庫的統計信息，包括向量數量、已處理文件等
    """
    try:
        return UpdateService.get_knowledge_status()
    except Exception as e:
        print(f"❌ 獲取知識庫信息失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取知識庫信息失敗: {str(e)}"
        )

@router.delete("/reset", summary="重置知識庫", description="危險操作！清除知識庫中的所有數據")
async def reset_knowledge_base():
    """
    重置知識庫（危險操作）
    
    警告：此操作會刪除向量數據庫中的所有數據，且無法恢復
    只在需要完全重建知識庫時使用
    """
    try:
        return {
            "message": "重置功能尚未實現，請手動清理",
            "warning": "這是危險操作，請謹慎使用"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置失敗: {str(e)}"
        )