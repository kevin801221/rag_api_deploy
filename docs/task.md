# RAG 向量資料庫完整應用指南

## 一、RAG 向量資料庫建置流程

### 1.1 Qdrant 向量資料庫建置

**Qdrant Cloud 建置步驟：**

1. **建構 Cluster**
   - 前往 [Qdrant Cloud](https://cloud.qdrant.io)
   - 註冊帳號並建立新的 cluster

2. **建立 API Key**
   - 在 cluster 管理頁面產生 API key
   - 記錄端點和 API key 資訊

3. **連線資訊範例**
   ```python
   # 配置資訊
   QDRANT_URL = ""
   QDRANT_API_KEY = ""
   MONGODB_URI = ""
   # 連線設定
   from qdrant_client import QdrantClient
   
   client = QdrantClient(
       url=QDRANT_URL,
       api_key=QDRANT_API_KEY,
   )
   ```

**💡 補充資訊：**
- 免費版本提供 1GB 儲存空間
- 支援 1536 維向量（適用 OpenAI embeddings）
- 建議在生產環境使用 HTTPS 連線

### 1.2 其他向量資料庫選項

| 資料庫 | 特色 | 適用場景 |
|--------|------|----------|
| **Weaviate** | 強型別 schema，支援多模態 | 企業級應用 |
| **Pinecone** | 全託管，高效能 | 快速部署 |
| **Chroma** | 輕量級，易於本地開發 | 原型開發 |
| **Milvus** | 開源，可自部署 | 自建基礎設施 |

## 二、文件轉向量的完整步驟

### 2.1 文件載入 (Document Loading)

**支援格式與對應 Loader：**

```python
from langchain_community.document_loaders import (
    TextLoader,      # .txt
    PyPDFLoader,     # .pdf
    CSVLoader,       # .csv
    JSONLoader,      # .json
    UnstructuredXMLLoader,  # .xml
    UnstructuredHTMLLoader, # .html
    Docx2txtLoader,  # .docx
    UnstructuredPowerPointLoader,  # .ppt, .pptx
)

# 載入範例
loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

**📚 參考資源：** [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)

### 2.2 文件切段 (Text Splitting)

**四種主要切段策略：**

#### 1. 字元切段 (CharacterTextSplitter)
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)
```

#### 2. Token 切段 (TokenTextSplitter)
```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50
)
```

#### 3. 遞歸字元切段 (RecursiveCharacterTextSplitter)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=100
)
```

#### 4. 遞歸 Token 切段 (RecursiveTokenTextSplitter)
```python
from langchain.text_splitter import RecursiveTokenTextSplitter

splitter = RecursiveTokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50
)
```

**🎯 選擇建議：**
- **一般文檔**：RecursiveCharacterTextSplitter
- **程式碼**：語言特定的 RecursiveCharacterTextSplitter
- **結構化資料**：TokenTextSplitter
- **大型文檔**：RecursiveTokenTextSplitter

### 2.3 向量嵌入 (Embedding)

**支援的嵌入模型：**

```python
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Cohere Embeddings
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-multilingual-v2.0")

# HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sentence Transformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

**📊 模型比較：**

| 模型 | 維度 | 語言支援 | 效能 | 成本 |
|------|------|----------|------|------|
| text-embedding-3-small | 1536 | 多語言 | 高 | 中 |
| text-embedding-3-large | 3072 | 多語言 | 最高 | 高 |
| all-MiniLM-L6-v2 | 384 | 英文為主 | 中 | 免費 |

**📚 參考資源：** [LangChain Text Embeddings](https://python.langchain.com/docs/integrations/text_embedding/)

### 2.4 向量儲存 (Vector Storage)

```python
# Qdrant
from langchain_qdrant import Qdrant
vector_store = Qdrant.from_documents(
    documents, embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY
)

# Weaviate
from langchain_weaviate import WeaviateVectorStore
vector_store = WeaviateVectorStore.from_documents(documents, embeddings)
```

**📚 參考資源：** [LangChain VectorStores](https://python.langchain.com/docs/concepts/vectorstores/)

## 三、切段與 Metadata 最佳策略

### 3.1 環境準備

**必要套件安裝：**
```bash
pip install -qU langchain langchain-openai langchain-mongodb langchain-experimental ragas pymongo tqdm
```

**連線設定：**
```python
import getpass
import os
from openai import OpenAI

# MongoDB 連線
MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")

# OpenAI 連線
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
openai_client = OpenAI()
```

### 3.2 資料載入與檢視

```python
from langchain_community.document_loaders import WebBaseLoader

# 載入測試資料
web_loader = WebBaseLoader([
    "https://peps.python.org/pep-0483/",
    "https://peps.python.org/pep-0008/",
    "https://peps.python.org/pep-0257/",
])
pages = web_loader.load()

# Document 結構範例
print("Document 結構：")
print(f"page_content: {pages[0].page_content[:100]}...")
print(f"metadata: {pages[0].metadata}")
```

**📋 Document 結構說明：**
- `page_content`：文檔內容
- `metadata`：包含 source、title、description、language 等資訊

### 3.3 切段策略函數定義

#### 固定 Token 切段
```python
from langchain.text_splitter import TokenTextSplitter
from typing import List, Optional

def fixed_token_split(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    固定 token 切段
    
    Args:
        docs: 文檔列表
        chunk_size: 切段大小（token 數量）
        chunk_overlap: 重疊 token 數量
    
    Returns:
        切段後的文檔列表
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
```

#### 遞歸切段
```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

def recursive_split(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    language: Optional[Language] = None,
) -> List[Document]:
    """
    遞歸切段
    
    Args:
        docs: 文檔列表
        chunk_size: 切段大小（token 數量）
        chunk_overlap: 重疊 token 數量
        language: 程式語言類型（可選）
    
    Returns:
        切段後的文檔列表
    """
    separators = ["\n\n", "\n", " ", ""]
    
    if language is not None:
        try:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)
        except (NameError, ValueError) as e:
            print(f"語言 {language} 無可用分隔符，使用預設值")
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)
```

#### 語義切段
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def semantic_split(docs: List[Document]) -> List[Document]:
    """
    語義切段
    
    Args:
        docs: 文檔列表
    
    Returns:
        語義切段後的文檔列表
    """
    splitter = SemanticChunker(
        OpenAIEmbeddings(), 
        breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(docs)
```

**🔧 語義切段參數選項：**
- `percentile`：使用 95% 分位數作為閾值
- `standard_deviation`：使用標準差方法
- `interquartile`：使用四分位距方法

### 3.4 評估資料集生成

```python
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 設定執行配置
RUN_CONFIG = RunConfig(max_workers=4, max_wait=180)

# 配置生成模型
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# 設定問題類型分佈
distributions = {
    simple: 0.5,        # 簡單問題 50%
    multi_context: 0.4, # 多語境問題 40%
    reasoning: 0.1      # 推理問題 10%
}

# 生成測試資料集
testset = generator.generate_with_langchain_docs(
    pages, 10, distributions, run_config=RUN_CONFIG
)
```

**📊 問題類型說明：**
- **simple**：直接從文檔中可以找到答案的問題
- **multi_context**：需要多個文檔片段才能回答的問題
- **reasoning**：需要推理和分析的複雜問題

### 3.5 MongoDB Atlas 向量儲存設定

```python
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# MongoDB 連線設定
client = MongoClient(MONGODB_URI)
DB_NAME = "evals"
COLLECTION_NAME = "chunking"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

def create_vector_store(docs: List[Document]) -> MongoDBAtlasVectorSearch:
    """
    建立 MongoDB Atlas 向量儲存
    
    Args:
        docs: 文檔列表
    
    Returns:
        MongoDB Atlas 向量儲存實例
    """
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return vector_store
```

**⚙️ MongoDB Atlas 索引配置：**
```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

### 3.6 切段策略評估

```python
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
import nest_asyncio

# 設定 asyncio
nest_asyncio.apply()
tqdm.get_lock().locks = []

# 準備評估資料
QUESTIONS = testset.question.to_list()
GROUND_TRUTH = testset.ground_truth.to_list()

def perform_eval(docs: List[Document]) -> Dict[str, float]:
    """
    執行 RAGAS 評估
    
    Args:
        docs: 切段後的文檔列表
    
    Returns:
        評估指標字典
    """
    eval_data = {
        "question": QUESTIONS,
        "ground_truth": GROUND_TRUTH,
        "contexts": [],
    }
    
    # 清空現有文檔
    print(f"清空集合 {DB_NAME}.{COLLECTION_NAME} 中的現有文檔")
    MONGODB_COLLECTION.delete_many({})
    print("清空完成")
    
    # 建立向量儲存
    vector_store = create_vector_store(docs)
    
    # 為評估資料集獲取相關文檔
    print("獲取評估集的語境")
    for question in tqdm(QUESTIONS):
        retrieved_docs = vector_store.similarity_search(question, k=3)
        eval_data["contexts"].append([doc.page_content for doc in retrieved_docs])
    
    # RAGAS 評估
    dataset = Dataset.from_dict(eval_data)
    print("執行評估")
    
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=RUN_CONFIG,
        raise_exceptions=False,
    )
    return result
```

### 3.7 批量評估執行

```python
# 評估不同切段策略
strategies_results = {}

for chunk_size in [100, 200, 500, 1000]:
    chunk_overlap = int(0.15 * chunk_size)  # 15% 重疊率
    
    print(f"\n======= 切段大小: {chunk_size} =======")
    
    # 1. 固定 token 無重疊
    print("------ 固定 token 無重疊 ------")
    result = perform_eval(fixed_token_split(pages, chunk_size, 0))
    print(f"結果: {result}")
    
    # 2. 固定 token 有重疊
    print("------ 固定 token 有重疊 ------")
    result = perform_eval(fixed_token_split(pages, chunk_size, chunk_overlap))
    print(f"結果: {result}")
    
    # 3. 遞歸切段有重疊
    print("------ 遞歸切段有重疊 ------")
    result = perform_eval(recursive_split(pages, chunk_size, chunk_overlap))
    print(f"結果: {result}")
    
    # 4. Python 特定遞歸切段
    print("------ Python 特定遞歸切段 ------")
    result = perform_eval(recursive_split(pages, chunk_size, chunk_overlap, Language.PYTHON))
    print(f"結果: {result}")

# 5. 語義切段
print("\n------ 語義切段 ------")
result = perform_eval(semantic_split(pages))
print(f"結果: {result}")
```

## 四、評估指標與結果分析

### 4.1 評估指標說明

**Context Precision（語境精確度）**
- **定義**：評估檢索器按相關性順序排列檢索項目的能力
- **計算**：相關檢索項目 / 總檢索項目
- **目標**：越接近 1.0 越好

**Context Recall（語境召回率）**
- **定義**：衡量檢索語境與真實答案的對齊程度
- **計算**：檢索到的相關資訊 / 所有相關資訊
- **目標**：越接近 1.0 越好

### 4.2 結果解讀範例

```python
# 範例評估結果
results_example = {
    "策略": ["固定Token無重疊", "固定Token有重疊", "遞歸切段", "Python特定", "語義切段"],
    "最佳切段大小": [500, 100, 100, 100, "N/A"],
    "Context Precision": [0.8833, 0.9, 0.9, 0.9833, 0.9],
    "Context Recall": [0.95, 0.95, 0.9833, 0.9833, 0.8187]
}

# 在此範例中，Python特定分割在兩個指標上都表現最佳
```

### 4.3 最佳化建議

**基於評估結果的策略選擇：**

1. **高精確度需求**：選擇 Context Precision 最高的策略
2. **高召回率需求**：選擇 Context Recall 最高的策略
3. **平衡考量**：計算 F1 分數 = 2 × (Precision × Recall) / (Precision + Recall)

**參數調優建議：**
```python
def optimize_parameters():
    """參數最佳化範例"""
    best_f1 = 0
    best_config = {}
    
    for size in [100, 200, 300, 500]:
        for overlap_ratio in [0.1, 0.15, 0.2]:
            overlap = int(size * overlap_ratio)
            
            # 執行評估
            chunks = recursive_split(pages, size, overlap, Language.PYTHON)
            results = perform_eval(chunks)
            
            # 計算 F1 分數
            precision = results['context_precision']
            recall = results['context_recall']
            f1 = 2 * (precision * recall) / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    'chunk_size': size,
                    'overlap': overlap,
                    'f1_score': f1
                }
    
    return best_config
```

## 五、進階 Metadata 設計策略

### 5.1 基礎 Metadata 結構

```python
def enhance_metadata(document, chunk_index, total_chunks):
    """增強 metadata 資訊"""
    enhanced_metadata = {
        # 原始 metadata
        **document.metadata,
        
        # 切段資訊
        "chunk_id": f"{document.metadata.get('source', 'unknown')}_{chunk_index}",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_size": len(document.page_content),
        
        # 內容分析
        "has_code": "```" in document.page_content or "def " in document.page_content,
        "has_table": "|" in document.page_content,
        "paragraph_count": document.page_content.count("\n\n"),
        
        # 時間戳記
        "indexed_at": datetime.now().isoformat(),
        
        # 內容摘要（可選）
        "summary": generate_summary(document.page_content) if len(document.page_content) > 1000 else None
    }
    
    return enhanced_metadata
```

### 5.2 動態 Metadata 生成

```python
def generate_dynamic_metadata(text):
    """動態生成 metadata"""
    import re
    from collections import Counter
    
    metadata = {
        # 基本統計
        "word_count": len(text.split()),
        "char_count": len(text),
        "sentence_count": len(re.split(r'[.!?]+', text)),
        
        # 內容特徵
        "has_numbers": bool(re.search(r'\d+', text)),
        "has_urls": bool(re.search(r'https?://', text)),
        "has_emails": bool(re.search(r'\S+@\S+', text)),
        
        # 語言檢測
        "language": detect_language(text),
        
        # 主題關鍵詞
        "keywords": extract_keywords(text, top_k=5),
        
        # 難度評估
        "reading_difficulty": assess_reading_difficulty(text)
    }
    
    return metadata
```

## 六、生產環境部署建議

### 6.1 效能監控

```python
import time
from functools import wraps

def monitor_performance(func):
    """效能監控裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} 執行時間: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

@monitor_performance
def chunking_pipeline(documents):
    """完整切段流水線"""
    # 執行切段邏輯
    pass
```

### 6.2 錯誤處理

```python
def robust_chunking(documents, strategy="recursive", **kwargs):
    """穩健的切段處理"""
    try:
        if strategy == "recursive":
            return recursive_split(documents, **kwargs)
        elif strategy == "semantic":
            return semantic_split(documents)
        elif strategy == "fixed":
            return fixed_token_split(documents, **kwargs)
        else:
            raise ValueError(f"不支援的策略: {strategy}")
            
    except Exception as e:
        print(f"切段失敗: {e}")
        # 回退到最簡單的策略
        return fixed_token_split(documents, chunk_size=500, chunk_overlap=50)
```

### 6.3 快取機制

```python
import hashlib
import pickle
import os

class ChunkingCache:
    """切段結果快取"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, documents, strategy, **kwargs):
        """生成快取鍵值"""
        content_hash = hashlib.md5(
            "".join([doc.page_content for doc in documents]).encode()
        ).hexdigest()
        
        params_hash = hashlib.md5(
            f"{strategy}_{kwargs}".encode()
        ).hexdigest()
        
        return f"{content_hash}_{params_hash}.pkl"
    
    def get(self, documents, strategy, **kwargs):
        """獲取快取結果"""
        cache_key = self._get_cache_key(documents, strategy, **kwargs)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, documents, strategy, result, **kwargs):
        """設定快取結果"""
        cache_key = self._get_cache_key(documents, strategy, **kwargs)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
```

## 總結

本指南涵蓋了 RAG 向量資料庫應用的完整流程，從基礎建置到進階最佳化。關鍵要點包括：

1. **選擇合適的向量資料庫**：根據需求選擇 Qdrant、Weaviate 或其他方案
2. **最佳化切段策略**：透過系統性評估找到最適合的方法
3. **設計有效的 Metadata**：提升檢索精確度和可維護性
4. **持續監控和最佳化**：建立評估機制確保系統效能

透過遵循這些最佳實踐，您可以建置出高效、可靠的 RAG 系統，為您的應用提供強大的語意搜尋能力。