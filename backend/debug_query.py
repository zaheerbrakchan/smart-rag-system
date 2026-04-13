"""Ad-hoc debug: query pgvector for NTA bulletin chunks (run: python debug_query.py)."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    from llama_index.core.vector_stores import VectorStoreQuery
    from llama_index.core.vector_stores.types import VectorStoreQueryMode
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
    from llama_index.embeddings.openai import OpenAIEmbedding
    from openai import OpenAI

    from services.vector_store_factory import get_vector_store

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vs = get_vector_store()
    embed = OpenAIEmbedding(model="text-embedding-3-small")

    query = "NEET exam"
    qv = embed.get_text_embedding(query)
    mf = MetadataFilters(
        filters=[
            MetadataFilter(
                key="document_type",
                value="nta_bulletin",
                operator=FilterOperator.EQ,
            )
        ]
    )
    vq = VectorStoreQuery(
        query_embedding=qv,
        similarity_top_k=50,
        filters=mf,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = await vs.aquery(vq)
    print(f"Got {len(result.nodes)} chunks")


if __name__ == "__main__":
    asyncio.run(main())
