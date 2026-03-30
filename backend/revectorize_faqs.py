"""
Quick script to revectorize all FAQs with the new embedding model
Run: python revectorize_faqs.py
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    from sqlalchemy import select, or_
    from pinecone import Pinecone
    from llama_index.embeddings.openai import OpenAIEmbedding
    import uuid
    
    from database.connection import async_session_maker
    from models.pending_qa import PendingQA, QAStatus
    
    # Initialize embedding model (same as documents)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc_index = pc.Index("neet-assistant")
    
    print("🔄 Starting FAQ re-vectorization...")
    print(f"   Using embedding model: text-embedding-3-small")
    
    async with async_session_maker() as db:
        # Get all approved/modified FAQs
        result = await db.execute(
            select(PendingQA).where(
                or_(PendingQA.status == QAStatus.APPROVED, PendingQA.status == QAStatus.MODIFIED)
            )
        )
        faqs = result.scalars().all()
        
        if not faqs:
            print("✅ No approved FAQs to revectorize")
            return
        
        print(f"   Found {len(faqs)} FAQs to revectorize")
        
        success_count = 0
        errors = []
        
        for faq in faqs:
            try:
                # Delete old vector if exists
                if faq.faq_vector_id:
                    try:
                        pc_index.delete(ids=[faq.faq_vector_id])
                        print(f"   🗑️  Deleted old vector: {faq.faq_vector_id}")
                    except Exception as e:
                        print(f"   ⚠️  Could not delete old vector: {e}")
                
                # Generate new embedding
                embedding = embed_model.get_text_embedding(faq.question)
                
                # Generate unique vector ID
                vector_id = f"faq_{faq.id}_{uuid.uuid4().hex[:8]}"
                
                # Prepare metadata
                metadata = {
                    "is_faq": True,
                    "faq_id": faq.id,
                    "question": faq.question,
                    "answer": faq.modified_answer or faq.original_answer,
                    "state": faq.detected_state or "All-India",
                    "exam": faq.detected_exam or "NEET",
                    "category": faq.detected_category or "general",
                    "document_type": "faq"
                }
                
                # Store in Pinecone
                pc_index.upsert(vectors=[(vector_id, embedding, metadata)])
                
                # Update database
                faq.faq_vector_id = vector_id
                
                success_count += 1
                print(f"   ✅ [{success_count}/{len(faqs)}] Revectorized FAQ {faq.id}: {faq.question[:50]}...")
                
            except Exception as e:
                errors.append({"faq_id": faq.id, "error": str(e)})
                print(f"   ❌ Error FAQ {faq.id}: {e}")
        
        await db.commit()
        
        print(f"\n🎉 Done! Revectorized {success_count}/{len(faqs)} FAQs")
        if errors:
            print(f"   ⚠️  Errors: {errors}")

if __name__ == "__main__":
    asyncio.run(main())
