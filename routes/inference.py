from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Tuple, Any
from core.raptor_core import ask_question

router = APIRouter()

class QueryRequest(BaseModel):
    question: str = Field(..., description="要提問的問題")
    knowledge_id: str = Field("rag_knowledge", description="知識庫 ID (對應 Qdrant collection name)")
    retrieval_k: int = Field(6, description="檢索的文檔片段數量")
    score_threshold: float = Field(None, description="檢索結果的最低分數閾值")

class Document(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Tuple[Document, float]] = Field(..., description="來源文檔及其相關性分數")

@router.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest = Body(...)
):
    """
    向知識庫提問並獲得答案和來源。
    """
    try:
        print(f"接收到問題: {request.question}")
        
        # 執行提問
        answer, relevant_docs = ask_question(
            question=request.question,
            retrieval_k=request.retrieval_k,
            score_threshold=request.score_threshold
        )
        
        if not answer and not relevant_docs:
            raise HTTPException(status_code=404, detail="無法從知識庫中找到相關資訊")
            
        # 格式化來源文檔
        formatted_docs = []
        for doc, score in relevant_docs:
            formatted_docs.append((
                Document(page_content=doc.page_content, metadata=doc.metadata),
                score
            ))
        
        return QueryResponse(answer=answer, source_documents=formatted_docs)
        
    except Exception as e:
        print(f"❌ 查詢時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {e}")