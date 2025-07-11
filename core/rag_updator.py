import os
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# 導入函數式 RAPTOR 核心
from .raptor_core import (
    # 初始化和配置
    init_raptor_core, get_config, update_config,
    # API Key 和模型設置
    load_api_keys_from_files, setup_models, setup_qdrant,
    # 文檔處理
    process_single_file_with_raptor, calculate_file_hash,
    # 向量存儲
    add_texts_to_vectorstore, build_rag_chain,
    # 統計和元數據
    get_vectorstore_stats, get_qdrant_file_metadata,
    # 高級便捷函數
    full_setup_raptor_system, get_current_state, reset_raptor_core
)

# 載入環境變量
load_dotenv()


# ===============================================
# 配置文件管理函數
# ===============================================

def load_updator_config(config_file: str = "rag_config.json") -> Dict:
    """載入或創建更新器配置文件
    
    Args:
        config_file: 配置文件路徑
        
    Returns:
        Dict: 配置字典
    """
    default_config = {
        "document_directory": "knowledge_docs",
        "last_update": None,
        "file_hashes": {},
        "chunk_size": 1500,
        "chunk_overlap": 150,
        "n_levels": 3,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "retrieval_k": 6,
        "batch_size": 20,
        "max_tokens_per_batch": 100000
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 合併新設定
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            print(f"✅ 載入配置文件: {config_file}")
        except Exception as e:
            print(f"⚠️ 讀取配置文件失敗，使用默認配置: {e}")
            config = default_config
    else:
        config = default_config
        print("📋 使用默認配置")
    
    return config


def save_updator_config(config: Dict, config_file: str = "rag_config.json"):
    """保存更新器配置文件
    
    Args:
        config: 配置字典
        config_file: 配置文件路徑
    """
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {config_file}")
    except Exception as e:
        print(f"⚠️ 保存配置文件失敗: {e}")


def check_config_changed(config: Dict, custom_config: Dict) -> bool:
    """檢查關鍵配置是否有變更
    
    Args:
        config: 當前配置
        custom_config: 自定義配置
        
    Returns:
        bool: 是否有配置變更
    """
    if not custom_config:
        return False
        
    key_params = ['chunk_size', 'chunk_overlap', 'n_levels']
    changed = False
    
    for param in key_params:
        if param in custom_config:
            old_value = config.get(param)
            new_value = custom_config[param]
            if old_value != new_value:
                print(f"🔄 配置變更: {param} = {old_value} → {new_value}")
                changed = True
                
    return changed


# ===============================================
# 系統初始化函數
# ===============================================

def initialize_raptor_system(config: Dict) -> bool:
    """初始化 RAPTOR 系統
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 初始化是否成功
    """
    try:
        # 載入 API Keys
        api_keys = load_api_keys_from_files()
        if not api_keys:
            print("❌ 找不到任何 OpenAI API Key！")
            return False
        
        # 初始化 RAPTOR 核心
        init_raptor_core(config)
        
        # 設置模型
        if not setup_models(openai_api_keys=api_keys):
            return False
        
        # 設置 Qdrant
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION", "rag_knowledge")
        
        if not qdrant_url:
            print("❌ 缺少 Qdrant URL 配置")
            return False
        
        # 如果不是本地端，則檢查 API Key
        if "localhost" not in qdrant_url and "127.0.0.1" not in qdrant_url:
            if not qdrant_api_key:
                print("❌ 遠程 Qdrant 需要 API Key")
                return False
        
        return setup_qdrant(qdrant_url, qdrant_api_key, collection_name)
        
    except Exception as e:
        print(f"❌ RAPTOR 系統初始化失敗: {e}")
        return False


# ===============================================
# 文件變更檢測函數
# ===============================================

def scan_directory_files(directory: str) -> List[str]:
    """掃描目錄中的支援文件
    
    Args:
        directory: 目錄路徑
        
    Returns:
        List[str]: 文件路徑列表
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        return []
    
    current_files = []
    for ext in ['*.pdf', '*.txt', '*.docx']:
        current_files.extend(directory_path.rglob(ext))
    
    return [str(f) for f in current_files]


def check_files_changed(config: Dict,
                       target_files: Optional[List[str]] = None,
                       config_changed: bool = False) -> Tuple[bool, List[str], List[str]]:
    """檢查文件是否有變更
    
    Args:
        config: 配置字典
        target_files: 指定要檢查的文件列表
        config_changed: 配置是否有變更
        
    Returns:
        Tuple[bool, List[str], List[str]]: (是否有變更, 新文件列表, 修改文件列表)
    """
    directory = config['document_directory']
    
    # 獲取文件列表
    if target_files:
        directory_path = Path(directory)
        current_files = []
        for filename in target_files:
            file_path = directory_path / filename
            if file_path.exists():
                current_files.append(str(file_path))
            else:
                print(f"⚠️ 找不到指定文件: {filename}")
        
        if not current_files:
            print("❌ 沒有找到任何指定的文件")
            return False, [], []
    else:
        current_files = scan_directory_files(directory)
    
    if not current_files:
        print(f"📂 目錄 {directory} 中沒有找到支援的文件")
        return False, [], []
    
    # 獲取 Qdrant 中已存儲的文件
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_knowledge")
    qdrant_files = get_qdrant_file_metadata(collection_name)
    
    new_files = []
    changed_files = []
    current_hashes = {}
    
    for file_path in current_files:
        current_hash = calculate_file_hash(file_path)
        current_hashes[file_path] = current_hash
        
        # 檢查本地配置記錄
        in_local_config = file_path in config['file_hashes']
        local_hash_match = in_local_config and config['file_hashes'][file_path] == current_hash
        
        # 檢查 Qdrant 中的記錄
        in_qdrant = file_path in qdrant_files
        qdrant_hash_match = in_qdrant and qdrant_files[file_path] == current_hash
        
        # 詳細狀態檢查
        print(f"🔍 檢查文件: {file_path}")
        print(f"   本地配置: {'✅' if in_local_config else '❌'} | 哈希匹配: {'✅' if local_hash_match else '❌'}")
        print(f"   Qdrant:   {'✅' if in_qdrant else '❌'} | 哈希匹配: {'✅' if qdrant_hash_match else '❌'}")
        
        # 如果配置有變更，強制重新處理
        if config_changed:
            print(f"   🔄 配置變更，強制重新處理")
            changed_files.append(file_path)
        # 檢查邏輯：兩邊都要有記錄且哈希匹配才跳過
        elif in_local_config and local_hash_match and in_qdrant and qdrant_hash_match:
            print(f"   ➡️ 跳過: 已存在且未修改")
        else:
            # 需要處理的情況
            if not in_local_config and not in_qdrant:
                new_files.append(file_path)
                print(f"   ➡️ 標記為: 新文件")
            elif not in_qdrant or not qdrant_hash_match:
                changed_files.append(file_path)
                print(f"   ➡️ 標記為: 需要更新（Qdrant 中缺失或不匹配）")
            elif not in_local_config or not local_hash_match:
                changed_files.append(file_path)
                print(f"   ➡️ 標記為: 需要更新（本地配置缺失或不匹配）")
    
    # 更新本地配置中的哈希值
    for file_path, hash_value in current_hashes.items():
        config['file_hashes'][file_path] = hash_value
    
    has_changes = bool(new_files or changed_files)
    
    if has_changes:
        print(f"\n📁 發現變更: 新增 {len(new_files)} 個，修改 {len(changed_files)} 個文件")
    else:
        print(f"\n✅ 所有 {len(current_files)} 個文件都已處理且未修改")
    
    return has_changes, new_files, changed_files


# ===============================================
# 文件處理函數
# ===============================================

def process_single_file(file_path: str, config: Dict) -> bool:
    """處理單個文件
    
    Args:
        file_path: 文件路徑
        config: 配置字典
        
    Returns:
        bool: 處理是否成功
    """
    try:
        print(f"📄 處理文件: {file_path}")
        
        # 使用 RAPTOR 處理文件
        all_texts, file_hash = process_single_file_with_raptor(file_path)
        
        if all_texts:
            # 創建元數據
            metadata_list = [{
                'source': file_path,
                'file_hash': file_hash,
                'timestamp': datetime.now().isoformat()
            } for _ in all_texts]
            
            # 添加到向量存儲
            success = add_texts_to_vectorstore(all_texts, metadata_list)
            if success:
                # 更新配置中的文件哈希
                config['file_hashes'][file_path] = file_hash
                print(f"✅ 完成: {file_path}")
                return True
            else:
                print(f"❌ 向量存儲失敗: {file_path}")
                return False
        else:
            print(f"⚠️ 空文件: {file_path}")
            return True  # 空文件不算錯誤
        
    except Exception as e:
        print(f"❌ 處理失敗 {file_path}: {e}")
        return False


def process_file_list(files: List[str], config: Dict) -> bool:
    """處理文件列表
    
    Args:
        files: 文件路徑列表
        config: 配置字典
        
    Returns:
        bool: 處理是否成功
    """
    if not files:
        return True
    
    print(f"📄 處理 {len(files)} 個文件...")
    
    success_count = 0
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 處理: {Path(file_path).name}")
        if process_single_file(file_path, config):
            success_count += 1
    
    print(f"📊 處理結果: {success_count}/{len(files)} 個文件成功")
    return success_count == len(files)


# ===============================================
# 調試函數
# ===============================================

def debug_qdrant_structure():
    """調試 Qdrant 中的數據結構"""
    from .raptor_core import _global_state
    
    qdrant_client = _global_state.get('qdrant_client')
    if not qdrant_client:
        print("❌ Qdrant 客戶端未初始化")
        return
    
    try:
        collection_name = os.getenv("QDRANT_COLLECTION", "rag_knowledge")
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=3,  # 只看前 3 個點
            with_payload=True,
            with_vectors=False
        )
        
        print(f"\n🔍 Qdrant 數據結構調試 (集合: {collection_name})")
        print("=" * 50)
        
        for i, point in enumerate(points, 1):
            print(f"\n點 {i} (ID: {point.id}):")
            if point.payload:
                import json
                print("Payload 結構:")
                print(json.dumps(point.payload, indent=2, ensure_ascii=False, default=str))
            else:
                print("  無 payload")
        
    except Exception as e:
        print(f"❌ 調試失敗: {e}")
        import traceback
        traceback.print_exc()


# ===============================================
# 主要更新函數
# ===============================================

def update_knowledge_base(custom_config: Dict = None, 
                         target_files: List[str] = None,
                         config_file: str = "rag_config.json") -> str:
    """檢查並更新知識庫 - 主要函數
    
    Args:
        custom_config: 自定義配置
        target_files: 指定要處理的文件列表
        config_file: 配置文件路徑
        
    Returns:
        str: 更新結果描述
    """
    print("🔄 檢查知識庫")
    print("=" * 30)
    
    try:
        # 載入配置
        config = load_updator_config(config_file)
        
        # 檢查系統是否已初始化
        current_state = get_current_state()
        if not current_state.get('qdrant_connected'):
            print("🚨 系統未在主應用程式中正確初始化，嘗試重新初始化...")
            if not initialize_raptor_system(config):
                return "❌ 系統初始化失敗"
        else:
            print("✅ 使用主應用程式已初始化的系統狀態")
        
        # 檢查文件變更
        has_changes, new_files, changed_files = check_files_changed(
            config, target_files, config_changed
        )
        
        if not has_changes:
            # 確保 RAG 鏈存在
            build_rag_chain()
            save_updator_config(config, config_file)
            return "✅ 這些資料都做過 RAG 了！"
        
        # 處理變更文件
        files_to_process = new_files + changed_files
        success = process_file_list(files_to_process, config)
        
        if success:
            # 建立 RAG 鏈
            build_rag_chain()
            
            # 更新配置
            config['last_update'] = datetime.now().isoformat()
            save_updator_config(config, config_file)
            
            return f"✅ 已更新 {len(files_to_process)} 個文件的 RAG 索引"
        else:
            return "❌ 更新 RAG 索引失敗"
            
    except Exception as e:
        return f"❌ 更新失敗: {e}"


# ===============================================
# 便捷函數
# ===============================================

def quick_update() -> str:
    """快速更新 - 使用預設設定
    
    Returns:
        str: 更新結果
    """
    return update_knowledge_base()


def update_specific_files(filenames: List[str]) -> str:
    """更新指定文件
    
    Args:
        filenames: 文件名列表
        
    Returns:
        str: 更新結果
    """
    return update_knowledge_base(target_files=filenames)


def update_with_config(chunk_size: int = None,
                      chunk_overlap: int = None, 
                      n_levels: int = None,
                      **kwargs) -> str:
    """使用自定義配置更新
    
    Args:
        chunk_size: 文本塊大小
        chunk_overlap: 重疊大小
        n_levels: RAPTOR 層數
        **kwargs: 其他配置參數
        
    Returns:
        str: 更新結果
    """
    custom_config = {}
    
    if chunk_size is not None:
        custom_config['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        custom_config['chunk_overlap'] = chunk_overlap
    if n_levels is not None:
        custom_config['n_levels'] = n_levels
    
    custom_config.update(kwargs)
    
    return update_knowledge_base(custom_config)


# ===============================================
# 統計和狀態函數
# ===============================================

def show_system_status():
    """顯示系統狀態"""
    print("\n📊 系統狀態:")
    print("=" * 30)
    
    # 顯示 RAPTOR 核心狀態
    state = get_current_state()
    for key, value in state.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    # 顯示向量庫統計
    stats = get_vectorstore_stats()
    if stats:
        print(f"\n📈 向量庫統計:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # 顯示配置
    config = get_config()
    if config:
        print(f"\n⚙️ 當前配置:")
        key_configs = ['chunk_size', 'chunk_overlap', 'n_levels', 'retrieval_k']
        for key in key_configs:
            if key in config:
                print(f"   {key}: {config[key]}")


def get_file_status(directory: str = "knowledge_docs") -> Dict:
    """獲取文件狀態
    
    Args:
        directory: 文檔目錄
        
    Returns:
        Dict: 文件狀態信息
    """
    files = scan_directory_files(directory)
    config = load_updator_config()
    
    file_status = {
        'total_files': len(files),
        'processed_files': 0,
        'unprocessed_files': 0,
        'files_info': []
    }
    
    for file_path in files:
        current_hash = calculate_file_hash(file_path)
        in_config = file_path in config['file_hashes']
        hash_match = in_config and config['file_hashes'][file_path] == current_hash
        
        status = "已處理" if in_config and hash_match else "未處理"
        if status == "已處理":
            file_status['processed_files'] += 1
        else:
            file_status['unprocessed_files'] += 1
        
        file_status['files_info'].append({
            'path': file_path,
            'status': status,
            'hash': current_hash[:8] + "..." if current_hash else "無法計算"
        })
    
    return file_status


# ===============================================
# 命令行參數解析
# ===============================================

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="RAG 更新器 - 智能文件檢測與知識庫更新（函數式版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python rag_updator.py                                     # 更新所有文件
  python rag_updator.py --path file1.pdf file2.txt         # 只處理指定文件
  python rag_updator.py --chunk_size 1000                  # 自定義文本塊大小
  python rag_updator.py --path file1.pdf --chunk_size 800 --n_levels 2  # 組合使用
  python rag_updator.py --debug                            # 調試 Qdrant 結構
  python rag_updator.py --status                           # 顯示系統狀態
        """
    )
    
    parser.add_argument(
        '--path', 
        nargs='+',
        help='指定要處理的文件名稱（在 knowledge_docs 目錄下）'
    )
    
    parser.add_argument(
        '--chunk_size', 
        type=int,
        help='文本分塊大小 (預設: 1500)'
    )
    
    parser.add_argument(
        '--chunk_overlap', 
        type=int,
        help='文本分塊重疊大小 (預設: 150)'
    )
    
    parser.add_argument(
        '--n_levels', 
        type=int,
        help='RAPTOR 層數 (預設: 3)'
    )
    
    parser.add_argument(
        '--embedding_model', 
        type=str,
        help='嵌入模型 (預設: text-embedding-3-small)'
    )
    
    parser.add_argument(
        '--llm_model', 
        type=str,
        help='語言模型 (預設: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--retrieval_k', 
        type=int,
        help='檢索結果數量 (預設: 6)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='調試模式：顯示 Qdrant 數據結構'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='顯示系統和文件狀態'
    )
    
    return parser.parse_args()


# ===============================================
# 主程序
# ===============================================

if __name__ == "__main__":
    args = parse_arguments()

    if args.debug:
        print("🐛 進入調試模式...")
        config = load_updator_config()
        if initialize_raptor_system(config):
            debug_qdrant_structure()
        else:
            print("❌ 系統初始化失敗，無法進行調試")

    elif args.status:
        show_system_status()

    else:
        custom_config = {}
        if args.chunk_size is not None:
            custom_config['chunk_size'] = args.chunk_size
        if args.chunk_overlap is not None:
            custom_config['chunk_overlap'] = args.chunk_overlap
        if args.n_levels is not None:
            custom_config['n_levels'] = args.n_levels
        if args.embedding_model is not None:
            custom_config['embedding_model'] = args.embedding_model
        if args.llm_model is not None:
            custom_config['llm_model'] = args.llm_model
        if args.retrieval_k is not None:
            custom_config['retrieval_k'] = args.retrieval_k
        
        result = update_knowledge_base(
            custom_config=custom_config if custom_config else None,
            target_files=args.path
        )
        print(f"\n✨ 更新完成: {result}")