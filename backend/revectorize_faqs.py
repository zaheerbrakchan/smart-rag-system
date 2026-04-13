"""
Quick script to revectorize all FAQs with the current embedding model.
Run: python revectorize_faqs.py
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    from sqlalchemy import select, or_
    from database.connection import async_session_maker
    from models.pending_qa import PendingQA, QAStatus
    from routes.faq import vectorize_and_store_faq
    from services.vector_store_factory import get_vector_store

    print("🔄 Starting FAQ re-vectorization (pgvector)...")

    async with async_session_maker() as db:
        result = await db.execute(
            select(PendingQA).where(
                or_(PendingQA.status == QAStatus.APPROVED, PendingQA.status == QAStatus.MODIFIED)
            )
        )
        faqs = result.scalars().all()

        if not faqs:
            print("✅ No approved FAQs to revectorize")
            return

        vs = get_vector_store()
        success_count = 0
        errors = []

        for faq in faqs:
            try:
                if faq.faq_vector_id:
                    try:
                        vs.delete_nodes(node_ids=[faq.faq_vector_id])
                    except Exception as e:
                        print(f"   ⚠️  Could not delete old vector: {e}")
                vector_id = await vectorize_and_store_faq(faq)
                faq.faq_vector_id = vector_id
                success_count += 1
                print(f"   ✅ [{success_count}/{len(faqs)}] FAQ {faq.id}: {faq.question[:50]}...")
            except Exception as e:
                errors.append({"faq_id": faq.id, "error": str(e)})
                print(f"   ❌ Error FAQ {faq.id}: {e}")

        await db.commit()

        print(f"\n🎉 Done! Revectorized {success_count}/{len(faqs)} FAQs")
        if errors:
            print(f"   ⚠️  Errors: {errors}")


if __name__ == "__main__":
    asyncio.run(main())
