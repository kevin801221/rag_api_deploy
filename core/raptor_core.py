import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import tiktoken
import umap
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture

# LangChain 相關導入
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 嘗試導入可選的文檔載入器
try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


# 全域狀態字典
_global_state = {
    'config': None,
    'embd': None,
    'model': None,
    'qdrant_client': None,
    'vectorstore': None,
    'rag_chain': None,
    'api_keys': [],
    'current_key_index': 0
}

# 常數
RANDOM_SEED = 224


# ===============================================
# 配置管理函數
# ===============================================

def get_default_config() -> Dict:
    """獲取預設配置"""
    return {
        "chunk_size": 1500,
        "chunk_overlap": 150,
        "n_levels": 3,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "retrieval_k": 6,
        "max_tokens_per_batch": 100000
    }


def init_raptor_core(config: Dict = None):
    """初始化 RAPTOR 核心
    
    Args:
        config: 配置字典，包含模型設置、分塊參數等
    """
    global _global_state
    
    _global_state['config'] = config or get_default_config()
    _global_state['embd'] = None
    _global_state['model'] = None
    _global_state['qdrant_client'] = None
    _global_state['vectorstore'] = None
    _global_state['rag_chain'] = None
    _global_state['api_keys'] = []
    _global_state['current_key_index'] = 0
    
    print("🚀 RAPTOR Core 初始化完成")


def get_config() -> Dict:
    """獲取當前配置"""
    return _global_state.get('config', get_default_config())


def update_config(new_config: Dict):
    """更新配置
    
    Args:
        new_config: 新的配置字典
    """
    if _global_state['config']:
        _global_state['config'].update(new_config)
    else:
        _global_state['config'] = new_config
    print("✅ 配置已更新")


def print_config():
    """打印當前配置"""
    config = get_config()
    print("\n📋 RAPTOR Core 配置:")
    print("=" * 40)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 40)


# ===============================================
# API Key 管理函數
# ===============================================

def load_api_keys_from_files() -> List[str]:
    """從各種配置文件載入 API Keys
    
    Returns:
        List[str]: API Key 列表
    """
    api_keys = []
    
    # 從環境變量載入
    for i in range(1, 10):
        env_key = os.getenv(f"OPENAI_API_KEY_{i}")
        if env_key:
            api_keys.append(env_key)
    
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key and env_key not in api_keys:
        api_keys.append(env_key)
    
    # 從 JSON 文件載入
    if os.path.exists('api_keys.json'):
        try:
            with open('api_keys.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'openai_api_keys' in data:
                    for key in data['openai_api_keys']:
                        if key and key not in api_keys:
                            api_keys.append(key)
        except Exception as e:
            print(f"⚠️ 讀取 api_keys.json 失敗: {e}")
    
    # 從文本文件載入
    if os.path.exists('openai_api_keys.txt'):
        try:
            with open('openai_api_keys.txt', 'r', encoding='utf-8') as f:
                keys = [line.strip() for line in f.readlines() 
                       if line.strip() and not line.startswith('#')]
                for key in keys:
                    if key not in api_keys:
                        api_keys.append(key)
        except Exception as e:
            print(f"⚠️ 讀取 openai_api_keys.txt 失敗: {e}")
    
    # 去重
    unique_keys = []
    seen = set()
    for key in api_keys:
        if key and key.strip() and key not in seen:
            unique_keys.append(key)
            seen.add(key)
    
    return unique_keys


def rotate_api_key() -> bool:
    """輪調 API Key"""
    global _global_state
    
    api_keys = _global_state.get('api_keys', [])
    current_index = _global_state.get('current_key_index', 0)
    
    if len(api_keys) > 1:
        new_index = (current_index + 1) % len(api_keys)
        new_key = api_keys[new_index]
        os.environ["OPENAI_API_KEY"] = new_key
        
        try:
            config = get_config()
            _global_state['embd'] = OpenAIEmbeddings(model=config['embedding_model'])
            _global_state['current_key_index'] = new_index
            print(f"🔄 切換到 API Key #{new_index + 1}")
            return True
        except Exception as e:
            print(f"⚠️ 切換 API Key 失敗: {e}")
            return False
    return False

# 模型設置函數

def setup_models(openai_api_key: str = None, 
                openai_api_keys: List[str] = None) -> bool:
    """設置 OpenAI 模型
    
    Args:
        openai_api_key: 單個 API Key
        openai_api_keys: 多個 API Key 列表
        
    Returns:
        bool: 設置是否成功
    """
    global _global_state
    
    try:
        # 設置 API Keys
        if openai_api_keys:
            api_keys = [key for key in openai_api_keys if key and key.strip()]
            if api_keys:
                _global_state['api_keys'] = api_keys
                os.environ["OPENAI_API_KEY"] = api_keys[0]
                print(f"✅ 設置了 {len(api_keys)} 個 OpenAI API Keys，支援輪調")
            else:
                raise ValueError("所有 API Key 都是空的")
        elif openai_api_key:
            if openai_api_key and openai_api_key.strip():
                _global_state['api_keys'] = [openai_api_key.strip()]
                os.environ["OPENAI_API_KEY"] = openai_api_key.strip()
                print("✅ 設置了 1 個 OpenAI API Key")
            else:
                raise ValueError("API Key 是空的")
        else:
            raise ValueError("沒有提供 API Key")
        
        config = get_config()
        
        # 初始化嵌入模型
        _global_state['embd'] = OpenAIEmbeddings(model=config['embedding_model'])
        print(f"✅ 嵌入模型初始化成功: {config['embedding_model']}")
        
        # 初始化語言模型
        _global_state['model'] = ChatOpenAI(
            temperature=0,
            model=config['llm_model']
        )
        print(f"✅ 語言模型初始化成功: {config['llm_model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型設置失敗: {e}")
        return False

# Qdrant 設置函數

def setup_qdrant(qdrant_url: str,
                qdrant_api_key: str,
                collection_name: str = "rag_knowledge",
                force_recreate: bool = False) -> bool:
    """設置 Qdrant 向量資料庫
    
    Args:
        qdrant_url: Qdrant 服務 URL
        qdrant_api_key: Qdrant API Key
        collection_name: 集合名稱
        force_recreate: 是否強制重建集合
        
    Returns:
        bool: 設置是否成功
    """
    global _global_state
    
    try:
        # 連接到 Qdrant
        _global_state['qdrant_client'] = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        print(f"✅ 成功連接到 Qdrant: {qdrant_url}")
        
        # 檢查集合是否存在
        collections = _global_state['qdrant_client'].get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if force_recreate and collection_exists:
            print(f"🗑️ 刪除現有集合: {collection_name}")
            _global_state['qdrant_client'].delete_collection(collection_name)
            collection_exists = False
        
        # 創建集合（如果不存在）
        if not collection_exists:
            print(f"🆕 創建新集合: {collection_name}")
            _global_state['qdrant_client'].create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small 的維度
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"📋 使用現有集合: {collection_name}")
        
        # 初始化向量存儲
        _global_state['vectorstore'] = QdrantVectorStore(
            client=_global_state['qdrant_client'],
            collection_name=collection_name,
            embedding=_global_state['embd']
        )
        print("✅ Qdrant 向量資料庫設置完成")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant 設置失敗: {e}")
        return False

# Token 計算函數

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """計算字串的 token 數量"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception:
        # 如果 tiktoken 失敗，使用粗略估計
        return int(len(string.split()) * 1.3)

# RAPTOR 聚類算法函數

def global_cluster_embeddings(embeddings: np.ndarray, 
                             dim: int,
                             n_neighbors: Optional[int] = None,
                             metric: str = "cosine") -> np.ndarray:
    """全局聚類嵌入"""
    try:
        if n_neighbors is None:
            n_neighbors = max(2, min(50, int((len(embeddings) - 1) ** 0.5)))
        
        return umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=dim, 
            metric=metric,
            random_state=RANDOM_SEED
        ).fit_transform(embeddings)
    except Exception as e:
        print(f"⚠️ 全局聚類失敗: {e}")
        return embeddings[:, :dim] if embeddings.shape[1] >= dim else embeddings

def local_cluster_embeddings(embeddings: np.ndarray,
                            dim: int,
                            num_neighbors: int = 10,
                            metric: str = "cosine") -> np.ndarray:
    """局部聚類嵌入"""
    try:
        num_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
        return umap.UMAP(
            n_neighbors=num_neighbors, 
            n_components=dim, 
            metric=metric,
            random_state=RANDOM_SEED
        ).fit_transform(embeddings)
    except Exception as e:
        print(f"⚠️ 局部聚類失敗: {e}")
        return embeddings[:, :dim] if embeddings.shape[1] >= dim else embeddings

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 20) -> int:
    """使用 BIC 獲取最佳聚類數量"""
    try:
        max_clusters = min(max_clusters, len(embeddings), 20)
        if max_clusters <= 1:
            return 1
            
        n_clusters = np.arange(1, max_clusters + 1)
        bics = []
        for n in n_clusters:
            try:
                gm = GaussianMixture(n_components=n, random_state=RANDOM_SEED)
                gm.fit(embeddings)
                bics.append(gm.bic(embeddings))
            except Exception:
                bics.append(float('inf'))
        
        return n_clusters[np.argmin(bics)]
    except Exception:
        return min(3, len(embeddings))

def GMM_cluster(embeddings: np.ndarray, threshold: float = 0.1):
    """高斯混合模型聚類"""
    try:
        n_clusters = get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
    except Exception as e:
        print(f"⚠️ GMM 聚類失敗: {e}")
        n_clusters = min(3, len(embeddings))
        labels = [np.array([i % n_clusters]) for i in range(len(embeddings))]
        return labels, n_clusters

def perform_clustering(embeddings: np.ndarray, 
                      dim: int = 10, 
                      threshold: float = 0.1) -> List[np.ndarray]:
    """執行聚類"""
    try:
        if len(embeddings) <= 2:
            return [np.array([0]) for _ in range(len(embeddings))]
        
        reduced_embeddings = global_cluster_embeddings(
            embeddings, 
            min(dim, embeddings.shape[1])
        )
        labels, n_clusters = GMM_cluster(reduced_embeddings, threshold)
        
        return labels
    except Exception as e:
        print(f"⚠️ 聚類過程失敗: {e}")
        return [np.array([i % 3]) for i in range(len(embeddings))]

# 嵌入處理函數

def embed_batch_with_retry(texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """帶重試和 API Key 輪調的批次嵌入"""
    embd = _global_state.get('embd')
    if not embd:
        print("❌ 嵌入模型未初始化")
        return [np.zeros(1536).tolist() for _ in texts]
    
    for attempt in range(max_retries):
        try:
            embeddings = embd.embed_documents(texts)
            return embeddings
        
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ 嵌入失敗 (嘗試 {attempt + 1}/{max_retries}): {error_msg}")
            
            # 檢查是否是 token 限制錯誤
            if "max_tokens_per_request" in error_msg or "too long" in error_msg.lower():
                print("   📉 Token 數量過多，分割批次...")
                mid = len(texts) // 2
                if mid > 0:
                    part1 = embed_batch_with_retry(texts[:mid], max_retries - attempt)
                    part2 = embed_batch_with_retry(texts[mid:], max_retries - attempt)
                    return part1 + part2
                else:
                    print(f"  ❌ 跳過過大的文本")
                    return [np.zeros(1536).tolist()]
            
            # 檢查是否是 API 限制錯誤
            elif any(keyword in error_msg.lower() for keyword in ["rate_limit", "quota", "limit"]):
                if rotate_api_key():
                    time.sleep(2)
                    continue
            
            # 其他錯誤，等待後重試
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"   ⏱️ 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
    
    print(f"   ❌ 批次嵌入最終失敗，返回零向量")
    return [np.zeros(1536).tolist() for _ in texts]


def embed_texts(texts: List[str]) -> np.ndarray:
    """將文本轉換為嵌入向量
    
    Args:
        texts: 文本列表
        
    Returns:
        np.ndarray: 嵌入向量數組
    """
    if not texts:
        return np.array([])
    
    config = get_config()
    all_embeddings = []
    current_batch = []
    current_tokens = 0
    max_tokens_per_batch = config['max_tokens_per_batch']
    
    print(f"📊 開始處理 {len(texts)} 個文本的嵌入...")
    
    for i, text in enumerate(texts):
        text_tokens = num_tokens_from_string(str(text))
        
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            print(f"   處理批次 (大小: {len(current_batch)}, tokens: {current_tokens:,})")
            batch_embeddings = embed_batch_with_retry(current_batch)
            all_embeddings.extend(batch_embeddings)
            
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
        
        if (i + 1) % 20 == 0:
            print(f"   進度: {i + 1}/{len(texts)}")
    
    # 處理最後一個批次
    if current_batch:
        print(f"   處理最後批次 (大小: {len(current_batch)}, tokens: {current_tokens:,})")
        batch_embeddings = embed_batch_with_retry(current_batch)
        all_embeddings.extend(batch_embeddings)
    
    print(f"✅ 嵌入處理完成，總共 {len(all_embeddings)} 個向量")
    return np.array(all_embeddings)


def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """嵌入並聚類文本"""
    try:
        text_embeddings_np = embed_texts(texts)
        if len(text_embeddings_np) == 0:
            return pd.DataFrame()
        
        cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
        
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        return df
    except Exception as e:
        print(f"⚠️ 嵌入聚類失敗: {e}")
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = [np.zeros(1536).tolist() for _ in texts]
        df["cluster"] = [np.array([0]) for _ in texts]
        return df

# 文本格式化和摘要函數

def fmt_txt(df: pd.DataFrame) -> str:
    """格式化文本"""
    if df.empty:
        return ""
    unique_txt = df["text"].tolist()
    return "\n\n".join(unique_txt)

def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """嵌入、聚類並摘要文本"""
    model = _global_state.get('model')
    if not model:
        print("❌ 語言模型未初始化")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        df_clusters = embed_cluster_texts(texts)
        
        if df_clusters.empty:
            print(f"--第 {level} 層處理失敗，返回空結果--")
            return pd.DataFrame(), pd.DataFrame()
        
        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append({
                    "text": row["text"],
                    "embd": row["embd"],
                    "cluster": cluster
                })
        
        if not expanded_list:
            print(f"--第 {level} 層沒有有效聚類--")
            return df_clusters, pd.DataFrame()
        
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()
        
        print(f"--第 {level} 層生成 {len(all_clusters)} 個聚類--")
        
        template = """這是一份文檔的子集。請為提供的文檔內容給出簡潔的摘要。

文檔內容:
{context}

摘要:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()
        
        summaries = []
        for i in all_clusters:
            try:
                df_cluster = expanded_df[expanded_df["cluster"] == i]
                formatted_txt = fmt_txt(df_cluster)
                if formatted_txt.strip():
                    summary = chain.invoke({"context": formatted_txt})
                    summaries.append(summary)
                else:
                    summaries.append(f"空聚類 {i}")
            except Exception as e:
                print(f"   ⚠️ 聚類 {i} 摘要失敗: {e}")
                summaries.append(f"聚類 {i} 摘要失敗")
        
        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })
        
        return df_clusters, df_summary
        
    except Exception as e:
        print(f"⚠️ 第 {level} 層處理失敗: {e}")
        return pd.DataFrame(), pd.DataFrame()

def recursive_embed_cluster_summarize(texts: List[str], level: int = 1) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """遞歸嵌入聚類摘要 - RAPTOR 核心算法"""
    config = get_config()
    results = {}
    
    try:
        df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)
        
        # 檢查是否繼續下一層
        if (level < config['n_levels'] and 
            not df_summary.empty and 
            df_summary["cluster"].nunique() > 1 and
            len(df_summary) > 1):
            
            new_texts = df_summary["summaries"].tolist()
            if new_texts and any(text.strip() for text in new_texts):
                next_level_results = recursive_embed_cluster_summarize(
                    new_texts, level + 1
                )
                results.update(next_level_results)
    
    except Exception as e:
        print(f"⚠️ 第 {level} 層遞歸處理失敗: {e}")
    
    return results

# 文檔載入函數

def load_documents_from_directory(directory_path: str) -> List:
    """從目錄載入各種格式的文檔"""
    documents = []
    
    # 確保目錄存在
    directory = Path(directory_path)
    if not directory.exists():
        print(f"📁 目錄不存在: {directory_path}")
        return []
    
    # 載入 PDF
    try:
        pdf_loader = DirectoryLoader(
            directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"📄 載入 {len(pdf_docs)} 個 PDF 文件")
    except Exception as e:
        print(f"⚠️ PDF 載入錯誤: {e}")
    
    # 載入文本文件
    try:
        txt_loader = DirectoryLoader(
            directory_path, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"📝 載入 {len(txt_docs)} 個 TXT 文件")
    except Exception as e:
        print(f"⚠️ TXT 載入錯誤: {e}")
    
    # 載入 Word 文件 (如果支援)
    if DOCX_SUPPORT:
        try:
            doc_loader = DirectoryLoader(
                directory_path, glob="**/*.docx",
                loader_cls=UnstructuredWordDocumentLoader
            )
            doc_docs = doc_loader.load()
            documents.extend(doc_docs)
            print(f"📄 載入 {len(doc_docs)} 個 DOCX 文件")
        except Exception as e:
            print(f"⚠️ DOCX 載入錯誤: {e}")
    
    return documents

def load_and_split_file(file_path: str) -> List[str]:
    """載入並分割單個文件"""
    config = get_config()
    
    try:
        # 根據文件類型選擇合適的載入器
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.docx') and DOCX_SUPPORT:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"⚠️ 不支持的文件類型: {file_path}")
            return []
        
        # 載入文檔
        docs = loader.load()
        if not docs:
            print(f"⚠️ 文件為空: {file_path}")
            return []
        
        # 合併所有頁面的內容
        all_content = []
        for doc in docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                all_content.append(doc.page_content)
        
        if not all_content:
            return []
        
        # 分割文檔
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        concatenated_content = "\n\n".join(all_content)
        chunks = text_splitter.split_text(concatenated_content)
        
        print(f"📄 從 {file_path} 分割出 {len(chunks)} 個片段")
        return chunks
        
    except Exception as e:
        print(f"⚠️ 載入文件失敗 {file_path}: {e}")
        return []

# RAPTOR 文檔處理函數

def process_documents_with_raptor(documents: List, file_hash: str = None) -> List[str]:
    """使用 RAPTOR 算法處理文檔
    
    Args:
        documents: 文檔列表
        file_hash: 文件哈希值（用於元數據）
        
    Returns:
        List[str]: 處理後的所有文本節點
    """
    config = get_config()
    print("\n🌳 開始 RAPTOR 處理...")
    
    # 1. 文檔內容合併
    all_content = []
    for doc in documents:
        if hasattr(doc, 'page_content') and doc.page_content:
            all_content.append(doc.page_content)
    
    if not all_content:
        print("❌ 文檔內容為空")
        return []
    
    concatenated_content = "\n\n\n --- \n\n\n".join(all_content)
    total_tokens = num_tokens_from_string(concatenated_content)
    print(f"📊 總 token 數: {total_tokens:,}")
    
    # 2. 文本分割
    print("✂️ 分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    texts_split = text_splitter.split_text(concatenated_content)
    print(f"📋 分割成 {len(texts_split)} 個文本塊")
    
    if not texts_split:
        print("❌ 文本分割結果為空")
        return []
    
    # 3. RAPTOR 樹構建
    print("🌳 建立 RAPTOR 樹結構...")
    all_texts = texts_split.copy()
    
    try:
        results = recursive_embed_cluster_summarize(texts_split, level=1)
        
        # 收集所有摘要
        for level in sorted(results.keys()):
            df_clusters, df_summary = results[level]
            if not df_summary.empty and 'summaries' in df_summary.columns:
                summaries = df_summary["summaries"].tolist()
                valid_summaries = [s for s in summaries if s and str(s).strip()]
                all_texts.extend(valid_summaries)
                print(f"   第 {level} 層添加 {len(valid_summaries)} 個摘要")
    
    except Exception as e:
        print(f"⚠️ RAPTOR 樹構建部分失敗: {e}")
        print("   將使用基本文本塊")
    
    print(f"📚 RAPTOR 處理完成，總共 {len(all_texts)} 個文本節點")
    return all_texts


def process_single_file_with_raptor(file_path: str) -> Tuple[List[str], str]:
    """使用 RAPTOR 處理單個文件
    
    Args:
        file_path: 文件路徑
        
    Returns:
        Tuple[List[str], str]: (處理後的文本列表, 文件哈希值)
    """
    print(f"\n📄 處理文件: {file_path}")
    
    # 計算文件哈希
    file_hash = calculate_file_hash(file_path)
    if not file_hash:
        return [], ""
    
    # 載入並分割文件
    text_chunks = load_and_split_file(file_path)
    if not text_chunks:
        return [], file_hash
    
    # 使用 RAPTOR 處理
    try:
        print("🌳 開始 RAPTOR 處理...")
        all_texts = text_chunks.copy()
        
        # 簡化的 RAPTOR 處理 - 只對較大的文件集合進行聚類
        if len(text_chunks) > 5:
            results = recursive_embed_cluster_summarize(text_chunks, level=1)
            
            # 收集摘要
            for level in sorted(results.keys()):
                df_clusters, df_summary = results[level]
                if not df_summary.empty and 'summaries' in df_summary.columns:
                    summaries = df_summary["summaries"].tolist()
                    valid_summaries = [s for s in summaries if s and str(s).strip()]
                    all_texts.extend(valid_summaries)
                    print(f"   第 {level} 層添加 {len(valid_summaries)} 個摘要")
        
        print(f"✅ RAPTOR 處理完成，總共 {len(all_texts)} 個文本節點")
        return all_texts, file_hash
        
    except Exception as e:
        print(f"⚠️ RAPTOR 處理失敗: {e}")
        return text_chunks, file_hash  # 回退到基本文本塊

# 向量存儲函數

def add_texts_to_vectorstore(texts: List[str], metadata_list: List[Dict] = None) -> bool:
    """將文本添加到向量存儲
    
    Args:
        texts: 文本列表
        metadata_list: 元數據列表
        
    Returns:
        bool: 是否成功
    """
    vectorstore = _global_state.get('vectorstore')
    if not vectorstore:
        print("❌ 向量存儲未初始化")
        return False
    
    if not texts:
        print("⚠️ 沒有文本需要添加")
        return True
    
    try:
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size] if metadata_list else None
            
            # 過濾空文本
            valid_indices = [j for j, text in enumerate(batch_texts) if text and str(text).strip()]
            valid_batch_texts = [batch_texts[j] for j in valid_indices]
            valid_batch_metadata = [batch_metadata[j] for j in valid_indices] if batch_metadata else None
            
            if valid_batch_texts:
                # 確保元數據格式正確（LangChain 會自動包裝到 metadata 字段中）
                if valid_batch_metadata:
                    # LangChain 的 add_texts 會自動將 metadatas 包裝到 payload.metadata 中
                    vectorstore.add_texts(valid_batch_texts, metadatas=valid_batch_metadata)
                else:
                    vectorstore.add_texts(valid_batch_texts)
                print(f"   已處理 {min(i+batch_size, len(texts))}/{len(texts)} 個文檔")
        
        print(f"✅ 成功添加 {len(texts)} 個文本到向量存儲")
        return True
        
    except Exception as e:
        print(f"❌ 添加文本到向量存儲失敗: {e}")
        return False

# RAG 查詢鏈函數

def build_rag_chain() -> bool:
    """建立 RAG 查詢鏈"""
    vectorstore = _global_state.get('vectorstore')
    model = _global_state.get('model')
    config = get_config()
    
    if not vectorstore:
        print("❌ 向量存儲未初始化")
        return False
    
    if not model:
        print("❌ 語言模型未初始化")
        return False
    
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config['retrieval_k']}
        )
        
        prompt = ChatPromptTemplate.from_template(
            """根據以下上下文回答問題。請提供詳細、準確的答案。如果信息不足，請明確指出。

上下文: {context}

問題: {question}

回答:"""
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        _global_state['rag_chain'] = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        
        print("✅ RAG 查詢鏈建立成功")
        return True
        
    except Exception as e:
        print(f"❌ RAG 查詢鏈建立失敗: {e}")
        return False


def ask_question(question: str, retrieval_k: int = None, score_threshold: float = None) -> Tuple[str, List[Tuple[Any, float]]]:
    """向 RAG 系統提問，並返回答案和相關文檔及分數
    
    Args:
        question: 問題
        retrieval_k: 檢索結果數量
        score_threshold: 檢索分數閾值
        
    Returns:
        Tuple[str, List[Tuple[Any, float]]]: (回答, 相關文檔及分數列表)
    """
    rag_chain = _global_state.get('rag_chain')
    vectorstore = _global_state.get('vectorstore')
    config = get_config()
    
    if not rag_chain:
        print("❌ RAG 查詢鏈未初始化，請先建立 RAG 鏈")
        return "", []
    
    print(f"❓ 問題: {question}")
    print("🤔 思考中...")
    
    try:
        # 獲取相關文檔及分數
        relevant_docs_with_scores = []
        if vectorstore:
            try:
                search_kwargs = {"k": retrieval_k if retrieval_k is not None else config['retrieval_k']}
                if score_threshold is not None:
                    search_kwargs["score_threshold"] = score_threshold
                
                relevant_docs_with_scores = vectorstore.similarity_search_with_score(
                    question,
                    **search_kwargs
                )
                print(f"📚 找到 {len(relevant_docs_with_scores)} 個相關文檔片段")
            except Exception as e:
                print(f"⚠️ 檢索相關文檔失敗: {e}")
        
        # 生成回答
        # 注意：rag_chain 預期的是一個 retriever，這裡需要調整
        # 為了保持 ask_question 的簡潔性，我們讓它直接調用 LLM
        # 並將相關文檔作為上下文傳入
        
        # 格式化相關文檔作為上下文
        context_docs = [doc for doc, score in relevant_docs_with_scores]
        formatted_context = "\n\n".join(doc.page_content for doc in context_docs)
        
        # 重新構建一個臨時的 chain 來處理這個問題，或者直接調用 LLM
        # 這裡為了簡化，直接使用 LLM 和一個簡單的 prompt
        model = _global_state.get('model')
        if not model:
            raise Exception("語言模型未初始化")
            
        prompt = ChatPromptTemplate.from_template(
            """根據以下上下文回答問題。請提供詳細、準確的答案。如果信息不足，請明確指出。

上下文: {context}

問題: {question}

回答:"""
        )
        
        chain = prompt | model | StrOutputParser()
        raw_answer = chain.invoke({"context": formatted_context, "question": question})
        
        print("💡 答案:")
        print("-" * 50)
        print(raw_answer)
        print("-" * 50)
        
        return raw_answer, relevant_docs_with_scores
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return "", []

# 統計和元數據函數

def get_vectorstore_stats() -> Dict:
    """獲取向量存儲統計信息"""
    qdrant_client = _global_state.get('qdrant_client')
    vectorstore = _global_state.get('vectorstore')
    
    if not qdrant_client or not vectorstore:
        return {}
    
    try:
        collection_name = vectorstore.collection_name
        collection_info = qdrant_client.get_collection(collection_name)
        
        return {
            "collection_name": collection_name,
            "total_vectors": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance
        }
    except Exception as e:
        print(f"⚠️ 獲取統計信息失敗: {e}")
        return {}


def get_qdrant_file_metadata(collection_name: str) -> Dict[str, str]:
    """從 Qdrant 獲取已存儲文件的元數據
    
    Args:
        collection_name: 集合名稱
        
    Returns:
        Dict[str, str]: 文件路徑到哈希值的映射
    """
    qdrant_client = _global_state.get('qdrant_client')
    if not qdrant_client:
        return {}
    
    try:
        # 查詢 Qdrant 中的所有點
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        # 提取文件元數據
        file_metadata = {}
        processed_count = 0
        
        for point in points:
            processed_count += 1
            
            # 檢查 payload 是否存在
            if not point.payload:
                continue
            
            # 檢查 LangChain 格式（metadata 在 payload.metadata 中）
            if 'metadata' in point.payload and point.payload['metadata'] is not None:
                metadata = point.payload['metadata']
                if isinstance(metadata, dict) and 'source' in metadata and 'file_hash' in metadata:
                    file_path = metadata['source']
                    file_hash = metadata['file_hash']
                    if file_path and file_hash:  # 確保不是空值
                        file_metadata[file_path] = file_hash
            
            # 檢查直接格式（metadata 直接在 payload 中）
            elif 'source' in point.payload and 'file_hash' in point.payload:
                file_path = point.payload['source']
                file_hash = point.payload['file_hash']
                if file_path and file_hash:  # 確保不是空值
                    file_metadata[file_path] = file_hash
        
        print(f"📋 從 Qdrant 處理了 {processed_count} 個點，找到 {len(file_metadata)} 個文件記錄")
        
        # 調試：顯示找到的文件
        if file_metadata:
            print("   找到的文件:")
            for file_path, file_hash in file_metadata.items():
                print(f"   - {file_path}: {file_hash[:8]}...")
        
        return file_metadata
        
    except Exception as e:
        print(f"⚠️ 獲取 Qdrant 元數據失敗: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calculate_file_hash(file_path: str) -> str:
    """計算文件的 MD5 哈希值
    
    Args:
        file_path: 文件路徑
        
    Returns:
        str: MD5 哈希值
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"⚠️ 無法計算文件哈希: {file_path}, 錯誤: {e}")
        return ""

# 高級便捷函數（保持與原類別的相容性）

def create_raptor_core(config: Dict = None, 
                      openai_api_key: str = None,
                      openai_api_keys: List[str] = None) -> bool:
    """創建並初始化 RAPTOR 核心
    
    Args:
        config: 配置字典
        openai_api_key: 單個 API Key
        openai_api_keys: 多個 API Key 列表
        
    Returns:
        bool: 初始化是否成功
    """
    # 初始化核心
    init_raptor_core(config)
    
    # 設置模型
    if openai_api_keys or openai_api_key:
        success = setup_models(
            openai_api_key=openai_api_key,
            openai_api_keys=openai_api_keys
        )
        if not success:
            print("❌ 模型設置失敗")
            return False
    
    return True


def get_current_state() -> Dict:
    """獲取當前全域狀態（用於除錯）"""
    return {
        'config_loaded': _global_state['config'] is not None,
        'embd_loaded': _global_state['embd'] is not None,
        'model_loaded': _global_state['model'] is not None,
        'qdrant_connected': _global_state['qdrant_client'] is not None,
        'vectorstore_ready': _global_state['vectorstore'] is not None,
        'rag_chain_built': _global_state['rag_chain'] is not None,
        'api_keys_count': len(_global_state['api_keys']),
        'current_key_index': _global_state['current_key_index']
    }


def reset_raptor_core():
    """重置 RAPTOR 核心狀態"""
    global _global_state
    _global_state = {
        'config': None,
        'embd': None,
        'model': None,
        'qdrant_client': None,
        'vectorstore': None,
        'rag_chain': None,
        'api_keys': [],
        'current_key_index': 0
    }
    print("🔄 RAPTOR Core 狀態已重置")

# 完整流程便捷函數

def full_setup_raptor_system(config: Dict = None,
                             qdrant_url: str = None,
                             qdrant_api_key: str = None,
                             collection_name: str = "rag_knowledge") -> bool:
    """完整設置 RAPTOR 系統
    
    Args:
        config: 配置字典
        qdrant_url: Qdrant URL（如果為 None 則從環境變數取得）
        qdrant_api_key: Qdrant API Key（如果為 None 則從環境變數取得）
        collection_name: 集合名稱
        
    Returns:
        bool: 設置是否成功
    """
    print("🚀 開始完整設置 RAPTOR 系統...")
    
    # 1. 載入 API Keys
    api_keys = load_api_keys_from_files()
    if not api_keys:
        print("❌ 找不到任何 OpenAI API Key！")
        return False
    
    # 2. 初始化核心
    if not create_raptor_core(config, openai_api_keys=api_keys):
        return False
    
    # 3. 設置 Qdrant
    if not qdrant_url:
        qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_api_key:
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ 缺少 Qdrant 配置")
        return False
    
    if not setup_qdrant(qdrant_url, qdrant_api_key, collection_name):
        return False
    
    # 4. 建立 RAG 鏈
    if not build_rag_chain():
        return False
    
    print("✅ RAPTOR 系統完整設置成功！")
    return True


def quick_process_and_ask(directory_path: str = "knowledge_docs",
                         question: str = "這些文檔的主要內容是什麼？",
                         show_sources: bool = True) -> str:
    """快速處理文檔並回答問題
    
    Args:
        directory_path: 文檔目錄
        question: 問題
        show_sources: 是否顯示來源
        
    Returns:
        str: 回答
    """
    # 載入並處理文檔
    documents = load_documents_from_directory(directory_path)
    if not documents:
        return "❌ 找不到任何文檔"
    
    # 使用 RAPTOR 處理
    all_texts = process_documents_with_raptor(documents)
    if not all_texts:
        return "❌ 文檔處理失敗"
    
    # 添加到向量存儲
    if not add_texts_to_vectorstore(all_texts):
        return "❌ 向量存儲添加失敗"
    
    # 重建 RAG 鏈
    if not build_rag_chain():
        return "❌ RAG 鏈建立失敗"
    
    # 回答問題
    return ask_question(question, show_sources)


# 使用範例

def example_usage():
    """使用範例"""
    print("🎯 RAPTOR Core 函數式版本使用範例")
    print("=" * 50)
    
    # 1. 完整設置系統
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "n_levels": 2,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "retrieval_k": 5
    }
    
    success = full_setup_raptor_system(config)
    if not success:
        print("❌ 系統設置失敗")
        return
    
    # 2. 處理文檔並回答問題
    answer = quick_process_and_ask(
        directory_path="knowledge_docs",
        question="這些文檔的主要內容是什麼？",
        show_sources=True
    )
    
    print(f"回答: {answer}")
    
    # 3. 顯示系統狀態
    state = get_current_state()
    print(f"\n系統狀態: {state}")


if __name__ == "__main__":
    example_usage()