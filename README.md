# RAPTOR RAG 系統

這是一個基於 RAPTOR (Recursive Abstractive Processing for Tree-structured Ontology Representation) 演算法的 RAG (Retrieval-Augmented Generation) 系統。它旨在提供高效的知識檢索和生成能力，並支援與 Dify 等平台的整合。

## 專案概覽

本專案提供了一個 FastAPI 服務，包含以下核心功能：

*   **RAPTOR 演算法**：用於處理非結構化文件，將其轉換為多層次的摘要和嵌入，以優化檢索效率和生成品質。
*   **向量資料庫整合**：使用 Qdrant 作為向量資料庫，儲存和管理知識庫的向量嵌入。
*   **Dify 相容的推論 API**：提供符合 Dify 平台規範的問答介面，方便與現有 LLM 應用整合。
*   **知識庫更新 API**：支援上傳新文件、更新現有文件，並自動將其處理並索引到向量資料庫中。

## 功能特色

*   **高效檢索**：RAPTOR 演算法能夠從大量文本中提取關鍵資訊，並以多層次結構儲存，提高檢索的精準度。
*   **靈活擴展**：基於 FastAPI 框架，易於部署和擴展。
*   **API 介面**：提供清晰的 RESTful API 介面，方便開發者整合。

## 環境設定

### 前置條件

*   Python 3.9+
*   pip (Python 套件管理器)
*   Qdrant 實例 (雲端或本地部署)
*   OpenAI API Key (用於嵌入和語言模型)

### 步驟

1.  **複製專案**

    ```bash
    git clone <你的專案 Git URL>
    cd RAG_poc
    ```

2.  **建立並啟用虛擬環境**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\\Scripts\\activate  # Windows
    ```

3.  **設定環境變數**

    在專案根目錄下建立一個 `.env` 檔案，並填入以下內容：

    ```dotenv
    # OpenAI API Keys - 每行一個，系統會自動輪調使用這些 Key 來避免速率限制
    OPENAI_API_KEY_1 = sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # OPENAI_API_KEY_2 = sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Qdrant 配置
    QDRANT_URL = <你的 Qdrant 服務 URL，例如：https://your-qdrant-instance.qdrant.tech:6333>
    QDRANT_API_KEY = <你的 Qdrant API Key>
    QDRANT_COLLECTION = <你的 Qdrant Collection 名稱，例如：qdrant_test>

    # 其他可選配置
    # DB_CONNECTION_STRING=mysql+pymysql://user:password@host:port/database_name
    # GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

    **重要提示**：請確保您的 `QDRANT_API_KEY` 具有足夠的權限來創建和管理指定的 `QDRANT_COLLECTION`。

4.  **安裝依賴**

    **使用 pip**：
    ```bash
    pip install -r requirements.txt # 如果有 requirements.txt
    # 或者手動安裝所有依賴 (請參考專案中的 setup.py 或直接安裝以下常用套件)
    pip install fastapi uvicorn python-dotenv pydantic qdrant-client langchain-community langchain-openai langchain-qdrant tiktoken umap-learn scikit-learn pandas pypdf
    ```

    **使用 uv (推薦)**：
    ```bash
    # 安裝 uv (如果尚未安裝)
    pip install uv
    
    # 創建並激活虛擬環境
    uv venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate  # Windows
    
    # 安裝依賴
    uv pip install -r requirements-uv.txt
    ```

    **注意**：使用 uv 可以獲得更快的依賴解析和安裝速度。

## 啟動服務

### 方法 1：直接使用 uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 方法 2：使用 run.py 腳本 (推薦)

```bash
# 基本啟動
python run.py

# 使用特定 RAG 方法
python run.py --rag-type raptor

# 啟用調試模式
python run.py --debug --reload

# 指定主機和端口
python run.py --host 127.0.0.1 --port 9000
```

**注意**：請確保您在專案根目錄下執行這些命令，並且 `main.py` 檔案存在。

服務將在指定地址上啟動（默認為 `http://0.0.0.0:8000`）。您可以透過瀏覽器訪問 `http://localhost:8000/docs` 查看 API 文件 (Swagger UI)。

## API 使用

### 1. 知識庫更新 API

用於上傳文件並將其處理後添加到向量資料庫。

*   **端點**：`/api/v1/knowledge/update` (支援文件上傳和異步處理)
*   **方法**：`POST`
*   **Content-Type**：`multipart/form-data`

**請求範例 (上傳文件)**：

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/update" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@/path/to/your/document.pdf" \
     -F "files=@/path/to/your/another_document.txt" \
     -F "async_processing=true"
```

**請求範例 (更新配置或處理指定文件，不含文件上傳)**：

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/update" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F 'config={"force_update": true, "chunk_size": 1000}' \
     -F 'target_files=["document1.pdf", "document2.txt"]'
```

*   **端點**：`/api/v1/knowledge/update-simple` (僅支援 JSON 請求，不支援文件上傳)
*   **方法**：`POST`
*   **Content-Type**：`application/json`

**請求範例 (簡化更新)**：

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/update-simple" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
           "config": {
             "chunk_size": 1000,
             "n_levels": 2,
             "force_update": true
           },
           "target_files": ["document1.pdf", "document2.txt"]
         }'
```

### 2. 推論 API (Dify 相容)

用於向知識庫提問並獲取相關答案和檢索到的記錄。

*   **端點**：`/api/v1/inference/ask`
*   **方法**：`POST`
*   **Content-Type**：`application/json`

**請求範例**：

```bash
curl -X POST "http://localhost:8000/api/v1/inference/ask" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
           "knowledge_id": "qdrant_test",
           "query": "你的知識庫主要內容是什麼？",
           "retrieval_setting": {
             "top_k": 5,
             "score_threshold": 0.5
           }
         }'
```

**響應範例**：

```json
{
  "records": [
    {
      "content": "文字片段",
      "score": 0.87,
      "title": "模索標題 (來源文件路徑)",
      "metadata": { "key": "value" }
    }
    // ... 更多記錄
  ]
}
```

### 3. 任務狀態查詢 API (異步更新)

如果使用異步更新，可以使用此端點查詢任務進度。

*   **端點**：`/api/v1/knowledge/status/{task_id}`
*   **方法**：`GET`

**請求範例**：

```bash
curl -X GET "http://localhost:8000/api/v1/knowledge/status/YOUR_TASK_ID" \
     -H "accept: application/json"
```

### 4. 知識庫資訊 API

獲取知識庫的統計資訊和狀態。

*   **端點**：`/api/v1/knowledge/info`
*   **方法**：`GET`

**請求範例**：

```bash
curl -X GET "http://localhost:8000/api/v1/knowledge/info" \
     -H "accept: application/json"
```

### 5. 健康檢查 API

檢查服務是否正常運行。

*   **端點**：`/api/v1/inference/health`
*   **方法**：`GET`

**請求範例**：

```bash
curl -X GET "http://localhost:8000/api/v1/inference/health" \
     -H "accept: application/json"
```

## 專案結構

### 目前結構

```
.
├─ core/
│   ├─ raptor_core.py        # RAPTOR 核心演算法實現
│   └─ rag_updator.py        # 知識庫更新邏輯
├─ models/
│   ├─ config_models.py      # 配置模型
│   ├─ request_models.py     # API 請求模型
│   └─ response_models.py    # API 響應模型
├─ routes/
│   ├─ inference.py          # 推論 API 端點
│   └─ update.py             # 知識庫更新 API 端點
├─ services/
│   ├─ inference_service.py  # 推論業務邏輯
│   ├─ raptor_service.py     # RAPTOR 系統初始化和狀態管理
│   └─ update_service.py     # 知識庫更新業務邏輯
├─ knowledge_docs/           # 存放原始知識文件 (PDF, TXT, DOCX)
├─ .env                      # 環境變數配置
├─ main.py                   # FastAPI 主應用程式入口
├─ run.py                    # 使用 uvicorn 啟動服務的腳本
├─ pyproject.toml            # 專案配置和依賴管理
├─ requirements-uv.txt       # uv 環境依賴清單
└─ README.md                 # 本文件
```

### 建議的模組化結構

為了支援多種 RAG 方法，建議將專案重構為以下結構：

```
.
├─ core/
│   ├─ common/              # 共用核心功能
│   │   ├─ embeddings.py    # 嵌入功能
│   │   ├─ vectorstore.py   # 向量存儲接口
│   │   └─ llm.py           # LLM 接口
│   ├─ raptor/              # RAPTOR 實現
│   │   ├─ core.py
│   │   └─ utils.py
│   ├─ advanced_rag/        # Advanced RAG 實現
│   │   ├─ core.py
│   │   └─ utils.py
│   └─ agentic_rag/         # Agentic RAG 實現
│       ├─ core.py
│       └─ utils.py
├─ models/
│   ├─ common/              # 共用模型
│   │   ├─ base_models.py
│   │   └─ config_models.py
│   ├─ raptor/              # RAPTOR 專用模型
│   ├─ advanced_rag/        # Advanced RAG 專用模型
│   └─ agentic_rag/         # Agentic RAG 專用模型
├─ routes/
│   ├─ common.py            # 共用路由
│   ├─ raptor.py            # RAPTOR 路由
│   ├─ advanced_rag.py      # Advanced RAG 路由
│   └─ agentic_rag.py       # Agentic RAG 路由
├─ services/
│   ├─ common/              # 共用服務
│   │   ├─ base_service.py
│   │   └─ vectorstore_service.py
│   ├─ raptor/              # RAPTOR 服務
│   ├─ advanced_rag/        # Advanced RAG 服務
│   └─ agentic_rag/         # Agentic RAG 服務
├─ knowledge_docs/           # 存放原始知識文件
├─ .env                      # 環境變數配置
├─ main.py                   # FastAPI 主應用程式入口
├─ run.py                    # 使用 uvicorn 啟動服務的腳本
├─ pyproject.toml            # 專案配置和依賴管理
├─ requirements-uv.txt       # uv 環境依賴清單
└─ README.md                 # 本文件