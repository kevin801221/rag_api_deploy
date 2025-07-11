# RAG-RAPTOR API 系統

這是一個基於 RAPTOR (Recursive Abstractive Processing for Tree-structured Ontology Representation) 演算法的進階 RAG (Retrieval-Augmented Generation) 系統。本專案透過 FastAPI 將整個服務打包，並使用 Docker 容器化，實現本地端的高效部署。

## 專案核心功能

*   **進階 RAG 演算法**：採用 RAPTOR 演算法，將文件遞歸地進行摘要和聚類，建立一個多層次的知識樹，以提升檢索的精準度和上下文的豐富性。
*   **本地向量資料庫**：整合 Qdrant 作為向量資料庫，所有資料（包含向量和索引）都儲存在本地的 `qdrant_storage` 目錄中，確保資料的持久性和私密性。
*   **容器化部署**：提供 `docker-compose.yml` 設定，一鍵啟動包含 FastAPI 應用和 Qdrant 資料庫在內的完整服務。
*   **動態知識庫更新**：提供 API 端點，可以動態地掃描 `knowledge_docs` 目錄下的新文件或修改過的舊文件，並自動將其處理、嵌入並索引到 Qdrant 資料庫中。
*   **簡單易用的推論 API**：提供清晰的 API 端點，用於對知識庫進行提問，並獲取由大型語言模型（LLM）生成的答案及相關的來源文件。

---

## 安裝與啟動

我們強烈建議使用 Docker 進行部署，因為它簡化了所有環境和服務的設定。

### 快速開始 (Docker - 推薦)

這種方式會同時啟動 FastAPI 應用程式和 Qdrant 向量資料庫兩個服務。

**前置條件:**
*   [Docker](https://www.docker.com/products/docker-desktop/) 已安裝並正在運行。
*   `docker-compose` 指令可用 (通常隨 Docker Desktop 一起安裝)。

**步驟:**

1.  **複製專案**
    ```bash
    git clone <你的專案 Git URL>
    cd RAG_poc
    ```

2.  **設定環境變數**
    在專案根目錄下，複製或重新命名 `.env.example` (如果有的話) 為 `.env`，或直接建立一個新的 `.env` 檔案。填入以下內容：

    ```dotenv
    # ================= OpenAI API Keys =================
    # 系統會自動輪調使用這些 Key，以避免單一 Key 的速率限制。
    # 至少需要提供一個。
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # OPENAI_API_KEY_2="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # OPENAI_API_KEY_3="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # ================= Qdrant 配置 =================
    # 這個 collection 名稱將在 Docker 環境中使用。
    # 當您啟動 Docker 時，如果這個 collection 不存在，系統會自動建立它。
    QDRANT_COLLECTION="qdrant_test"
    ```
    **重要提示**: `docker-compose.yml` 已被設定為會讀取此 `.env` 檔案中的 `QDRANT_COLLECTION` 變數。

3.  **建立並啟動 Docker 容器**
    在專案根目錄下，執行以下指令：
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`：會強制重新建置 Docker 映像檔，確保您的程式碼變更都已生效。
    *   `-d`：會在背景模式下執行容器。

4.  **確認服務狀態**
    ```bash
    docker-compose ps
    ```
    您應該會看到 `rag_api_service` 和 `qdrant_service` 兩個服務都處於 `running` 或 `up` 的狀態。

    服務啟動後，您可以透過瀏覽器訪問 `http://localhost:8000/docs` 來查看並互動式地測試 API。

### 本地開發設定 (不使用 Docker)

如果您希望直接在本地環境中執行，請遵循以下步驟。

**前置條件:**
*   Python 3.9+
*   `uv` (推薦) 或 `pip` 套件管理器。
*   一個正在運行的 Qdrant 實例 (您可以另外透過 Docker 或其他方式啟動)。

**步驟:**

1.  **複製專案並進入目錄** (同上)。

2.  **設定環境變數**
    建立 `.env` 檔案 (同上)，但您需要根據您的本地 Qdrant 設定來修改 `QDRANT_URL` 和 `QDRANT_API_KEY`。
    ```dotenv
    # ... (OpenAI Keys)

    # Qdrant 配置 (本地範例)
    QDRANT_URL="http://localhost:6333"
    QDRANT_API_KEY=  # 本地 Qdrant 通常不需要 API Key
    QDRANT_COLLECTION="qdrant_test"
    ```

3.  **安裝依賴 (使用 uv - 推薦)**
    ```bash
    # 安裝 uv (如果尚未安裝)
    pip install uv
    
    # 建立並激活虛擬環境
    uv venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\\Scripts\\activate  # Windows
    
    # 安裝依賴
    uv pip install -r requirements.txt
    ```

4.  **啟動服務**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

---

## API 使用說明

您可以透過 `http://localhost:8000/docs` 的互動式介面來測試以下 API，或使用 `curl` 等工具。

### 1. 更新知識庫

此 API 會掃描 `knowledge_docs` 目錄，並將新增或有變更的文件處理後加入到 Qdrant 資料庫中。

*   **端點**: `/api/update/knowledge`
*   **方法**: `POST`

**使用範例 (處理所有文件)**:
```bash
curl -X POST http://localhost:8000/api/update/knowledge \
     -H "Content-Type: application/json" \
     -d '{}'
```
這是一個背景任務，API 會立即回傳一個任務 ID。您可以透過日誌查看更新進度。

### 2. 查詢知識庫

向知識庫提問，並獲得由 RAG 系統生成的答案。

*   **端點**: `/api/inference/query`
*   **方法**: `POST`

**使用範例**:其實
```bash
curl -X POST http://localhost:8000/api/inference/query \
     -H "Content-Type: application/json" \
     -d '{
           "question": "CODE AGENT 是什麼？"
         }'
```

**響應範例**:其實
```json
{
  "answer": "CODE AGENT 是一種基於大型語言模型（LLM）的代碼生成框架...",
  "source_documents": [
    [
      {
        "page_content": "文檔介紹了CODE AGENT，一種基於大型語言模型的代碼生成框架...",
        "metadata": {
          "source": "knowledge_docs/2401.07339v2.pdf",
          "file_hash": "fdf2768f6632646890cfd06e20de56c0",
          "timestamp": "...",
          "_id": "...",
          "_collection_name": "qdrant_test"
        }
      },
      0.64846873
    ]
    // ... 更多來源文件
  ]
}
```

---

## 專案結構

```
.
├── core/
│   ├── raptor_core.py        # RAPTOR 核心演算法實現
│   └── rag_updator.py        # 知識庫更新邏輯
├── models/
│   ├── config_models.py      # 配置模型
│   ├── request_models.py     # API 請求模型
│   └── response_models.py    # API 響應模型
├── routes/
│   ├── inference.py          # 推論 API 端點
│   └── update.py             # 知識庫更新 API 端點
├── services/
│   ├── inference_service.py  # 推論業務邏輯
│   ├── raptor_service.py     # RAPTOR 系統初始化和狀態管理
│   └── update_service.py     # 知識庫更新業務邏輯
├── knowledge_docs/           # 存放原始知識文件 (PDF, TXT, DOCX)
├── qdrant_storage/           # Qdrant 本地資料庫儲存位置 (由 Docker 掛載)
├── .env                      # 環境變數配置
├── docker-compose.yml        # Docker 容器編排設定
├── Dockerfile                # FastAPI 應用的 Docker 映像檔設定
├── main.py                   # FastAPI 主應用程式入口
├── run.py                    # 使用 uvicorn 啟動服務的腳本
├── pyproject.toml            # 專案配置和依賴管理
├── requirements.txt          # Python 依賴清單
└── README.md                 # 本文件
```