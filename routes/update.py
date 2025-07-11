
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from core.rag_updator import update_knowledge_base

router = APIRouter()

class UpdateRequest(BaseModel):
    target_files: Optional[List[str]] = Field(None, description="指定要更新的文件列表。如果為空，則更新所有文件。")
    chunk_size: Optional[int] = Field(None, description="文本分塊大小")
    chunk_overlap: Optional[int] = Field(None, description="文本分塊重疊大小")
    n_levels: Optional[int] = Field(None, description="RAPTOR 樹的層數")

class UpdateResponse(BaseModel):
    message: str
    task_id: Optional[str] = None

# 簡單的背景任務管理器
background_tasks_status = {}

def run_update_in_background(task_id: str, config: dict, target_files: Optional[List[str]]):
    print(f"背景任務 {task_id} 開始...")
    background_tasks_status[task_id] = "running"
    try:
        result = update_knowledge_base(custom_config=config, target_files=target_files)
        background_tasks_status[task_id] = f"completed: {result}"
        print(f"背景任務 {task_id} 完成: {result}")
    except Exception as e:
        error_message = f"failed: {e}"
        background_tasks_status[task_id] = error_message
        print(f"背景任務 {task_id} 失敗: {error_message}")

@router.post("/knowledge", response_model=UpdateResponse)
async def update_knowledge(
    background_tasks: BackgroundTasks,
    request: UpdateRequest = Body(...)
):
    """
    觸發知識庫更新。這是一個長時間運行的任務，將在背景執行。
    """
    try:
        custom_config = {}
        if request.chunk_size is not None:
            custom_config['chunk_size'] = request.chunk_size
        if request.chunk_overlap is not None:
            custom_config['chunk_overlap'] = request.chunk_overlap
        if request.n_levels is not None:
            custom_config['n_levels'] = request.n_levels

        # 創建一個唯一的任務 ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # 將更新任務添加到背景
        background_tasks.add_task(
            run_update_in_background, 
            task_id, 
            custom_config, 
            request.target_files
        )
        
        background_tasks_status[task_id] = "pending"
        
        return UpdateResponse(
            message="知識庫更新已在背景開始。請使用 /status/{task_id} 查詢狀態。",
            task_id=task_id
        )

    except Exception as e:
        print(f"❌ 觸發更新時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {e}")

@router.get("/knowledge/status/{task_id}", response_model=UpdateResponse)
async def get_update_status(task_id: str):
    """
    查詢背景更新任務的狀態。
    """
    status = background_tasks_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="找不到指定的任務 ID")
    
    return UpdateResponse(message=status, task_id=task_id)

