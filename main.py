import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from routes import inference, update
from core.raptor_core import full_setup_raptor_system

# 在應用程式啟動前載入 .env 文件
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 應用程式啟動時執行的代碼
    print("🚀 應用程式啟動中...")
    
    # 從環境變數獲取 Qdrant 配置
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_knowledge")

    print(f"🔗 正在連接到 Qdrant: {qdrant_url}")
    
    # 執行完整的 RAPTOR 系統設置
    # 注意：這會載入模型，可能需要一些時間
    success = full_setup_raptor_system(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name
    )
    
    if success:
        print("✅ 系統初始化成功！")
    else:
        print("❌ 系統初始化失敗，請檢查日誌。")
        # 在生產環境中，您可能希望在這裡引發異常來停止啟動
        
    yield
    
    # 應用程式關閉時執行的代碼
    print("👋 應用程式關閉。")

app = FastAPI(
    title="RAG-RAPTOR API",
    description="一個使用 RAPTOR 算法和 Qdrant 的進階 RAG 系統",
    version="1.0.0",
    lifespan=lifespan
)

# 包含路由
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
app.include_router(update.router, prefix="/api/update", tags=["Knowledge Base"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "歡迎來到 RAG-RAPTOR API"}