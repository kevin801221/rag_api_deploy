import os
import sys
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from fastapi import UploadFile
import tempfile
import shutil

# 添加 core 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from rag_updator import (
    update_knowledge_base, load_updator_config, save_updator_config,
    get_file_status, show_system_status
)
from raptor_core import get_vectorstore_stats, reset_raptor_core
from .raptor_service import raptor_service

# 全局任務狀態存儲（生產環境應該使用 Redis 或數據庫）
task_status = {}

class UpdateService:
    """知識庫更新服務"""
    
    @staticmethod
    def get_knowledge_docs_dir() -> str:
        """獲取知識文檔目錄"""
        return os.getenv("KNOWLEDGE_DOCS_DIR", "knowledge_docs")
    
    @staticmethod
    async def save_uploaded_files(files: List[UploadFile]) -> List[str]:
        """保存上傳的文件到知識文檔目錄"""
        docs_dir = Path(UpdateService.get_knowledge_docs_dir())
        docs_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for file in files:
            if not file.filename:
                continue
                
            # 檢查文件類型
            allowed_extensions = {'.pdf', '.txt', '.docx'}
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_extensions:
                raise ValueError(f"不支持的文件類型: {file.filename}，支持的類型: {', '.join(allowed_extensions)}")
            
            # 保存文件
            file_path = docs_dir / file.filename
            
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                saved_files.append(file.filename)
                print(f"✅ 已保存文件: {file.filename}")
                
            except Exception as e:
                print(f"❌ 保存文件失敗 {file.filename}: {e}")
                raise
        
        return saved_files
    
    @staticmethod
    def create_task_id() -> str:
        """創建任務ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def update_task_status(task_id: str, status: str, progress: float = None, 
                          message: str = "", result: Dict = None):
        """更新任務狀態"""
        task_status[task_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "result": result,
            "timestamp": time.time()
        }
    
    @staticmethod
    def get_task_status(task_id: str) -> Optional[Dict]:
        """獲取任務狀態"""
        return task_status.get(task_id)
    
    @staticmethod
    def process_knowledge_update_sync(config: Optional[Dict] = None,
                                    target_files: Optional[List[str]] = None) -> Dict:
        """同步處理知識庫更新"""
        start_time = time.time()
        
        try:
            print("🔄 開始更新知識庫...")
            
            # 重置 RAPTOR 核心狀態，確保使用新配置
            reset_raptor_core()
            
            # 執行更新
            result_message = update_knowledge_base(
                custom_config=config,
                target_files=target_files
            )
            
            # 獲取統計信息
            stats = get_vectorstore_stats()
            
            # 分析處理結果
            if "已更新" in result_message or "完成" in result_message:
                success = True
                # 嘗試從結果信息中提取處理的文件數量
                import re
                file_count_match = re.search(r'(\d+)\s*個文件', result_message)
                processed_count = int(file_count_match.group(1)) if file_count_match else 0
            elif "都做過 RAG 了" in result_message:
                success = True
                processed_count = 0
            else:
                success = False
                processed_count = 0
            
            processing_time = time.time() - start_time
            
            # 重新初始化推論系統
            raptor_service.initialize_system(config)
            
            return {
                "success": success,
                "message": result_message,
                "total_files_processed": processed_count,
                "total_vectors_in_db": stats.get('total_vectors', 0),
                "processing_time": round(processing_time, 3),
                "config_used": config
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ 更新失敗: {error_msg}")
            
            return {
                "success": False,
                "message": f"更新失敗: {error_msg}",
                "total_files_processed": 0,
                "total_vectors_in_db": 0,
                "processing_time": round(processing_time, 3),
                "config_used": config
            }
    
    @staticmethod
    async def process_knowledge_update_async(task_id: str,
                                           config: Optional[Dict] = None,
                                           target_files: Optional[List[str]] = None):
        """異步處理知識庫更新"""
        try:
            # 更新任務狀態
            UpdateService.update_task_status(task_id, "processing", 0.1, "開始處理...")
            
            # 在線程池中執行同步處理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                UpdateService.process_knowledge_update_sync,
                config,
                target_files
            )
            
            # 更新完成狀態
            if result["success"]:
                UpdateService.update_task_status(
                    task_id, "completed", 1.0, "處理完成", result
                )
            else:
                UpdateService.update_task_status(
                    task_id, "failed", 1.0, result["message"], result
                )
                
        except Exception as e:
            UpdateService.update_task_status(
                task_id, "failed", 1.0, f"處理失敗: {str(e)}"
            )
    
    @staticmethod
    def get_knowledge_status() -> Dict:
        """獲取知識庫狀態"""
        try:
            docs_dir = UpdateService.get_knowledge_docs_dir()
            file_status = get_file_status(docs_dir)
            stats = get_vectorstore_stats()
            
            return {
                "knowledge_docs_dir": docs_dir,
                "total_files": file_status['total_files'],
                "processed_files": file_status['processed_files'],
                "unprocessed_files": file_status['unprocessed_files'],
                "total_vectors": stats.get('total_vectors', 0),
                "collection_name": stats.get('collection_name', 'unknown'),
                "files_info": file_status['files_info'][:10]  # 只返回前10個文件信息
            }
        except Exception as e:
            return {
                "error": f"獲取狀態失敗: {str(e)}"
            }