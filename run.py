#!/usr/bin/env python
"""
RAG POC API Server Runner
使用 uvicorn 運行 FastAPI 應用
"""
import os
import argparse
import uvicorn
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

def main():
    """解析命令行參數並啟動服務器"""
    parser = argparse.ArgumentParser(description="RAG POC API Server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="服務器監聽地址 (默認: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="服務器監聽端口 (默認: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="啟用代碼熱重載 (開發模式)"
    )
    parser.add_argument(
        "--rag-type", 
        type=str, 
        default="raptor", 
        choices=["raptor", "advanced_rag", "agentic_rag"],
        help="要使用的 RAG 方法 (默認: raptor)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="啟用調試模式"
    )
    
    args = parser.parse_args()
    
    # 設置環境變量
    os.environ["RAG_TYPE"] = args.rag_type
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    # 啟動服務器
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
    )

if __name__ == "__main__":
    main()
