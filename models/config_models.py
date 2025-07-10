from pydantic import BaseModel
from typing import Optional

class RaptorConfig(BaseModel):
    chunk_size: int = 1500
    chunk_overlap: int = 150
    n_levels: int = 3
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    retrieval_k: int = 6
    max_tokens_per_batch: int = 100000