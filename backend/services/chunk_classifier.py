"""
Chunk Classifier Service
Classifies document chunks into proper categories during indexing.

Naming (do not conflate with admin upload metadata):
- **doc_topic** (set on the Document at upload/reindex): the admin "sub-category" for the
  *whole file* (e.g. fees & payments vs comprehensive) — stored in vector metadata as ``doc_topic``.
- **chunk_category** / **chunk_section** / **chunk_importance** (set here): *page/chunk-level*
  labels describing what this slice of text is about (exam_info, seat_matrix, …).

Semantic search still uses embeddings; these labels help citations, analytics, and optional filters.
"""

import os
import re
from typing import Dict, Optional, Tuple
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============== CATEGORY DEFINITIONS ==============

# Categories for NTA Bulletin chunks
# NOTE: These categories are for metadata/display purposes only - NOT used for retrieval filtering.
# Semantic search handles finding relevant content regardless of category.
NTA_BULLETIN_CATEGORIES = {
    "exam_info": {
        "description": "Exam syllabus, pattern, dates, admit card, exam centers",
        "keywords": [
            "syllabus", "pattern", "marking", "negative marking", "questions",
            "exam date", "date of examination", "exam time", "duration", "admit card", "hall ticket",
            "exam center", "test center", "exam city", "shift", "session",
            "physics", "chemistry", "biology", "botany", "zoology",
            "mcq", "multiple choice", "correct answer", "wrong answer",
            "important dates", "at a glance", "events dates", "schedule",
            "timing of examination", "duration of examination"
        ]
    },
    "eligibility": {
        "description": "Age limit, educational qualification, attempt limits",
        "keywords": [
            "eligibility", "eligible", "age limit", "minimum age", "maximum age",
            "qualification", "12th", "class 12", "10+2", "intermediate",
            "pcb", "physics chemistry biology", "percentage", "marks required",
            "appearing candidate", "pass", "passed", "attempt"
        ]
    },
    "application": {
        "description": "Application fee, form filling, documents for application",
        "keywords": [
            "application", "apply", "registration", "application fee", "form",
            "online application", "application form", "payment", "pay fee",
            "photograph", "signature", "thumb impression", "category certificate",
            "application window", "correction window", "edit", "modify",
            "online submission", "fee payable", "last date", "submission of application"
        ]
    },
    "counselling": {
        "description": "MCC, AIQ, AIIMS, JIPMER, central quota counselling",
        "keywords": [
            "mcc", "medical counselling committee", "aiq", "all india quota",
            "15%", "15 percent", "aiims", "jipmer", "bhu", "amu", "du",
            "central university", "deemed university", "esic", "afmc",
            "central counselling", "central pool", "seat allotment"
        ]
    },
    "reservation": {
        "description": "Central level reservation categories",
        "keywords": [
            "reservation policy", "reserved category", "obc", "sc", "st", "ews", "pwd", "ph",
            "other backward class", "scheduled caste", "scheduled tribe",
            "economically weaker section", "person with disability", "benchmark disability",
            "27%", "7.5%", "horizontal reservation", "vertical reservation"
        ]
    },
    "result": {
        "description": "Merit list, qualifying marks, rank, scorecard",
        "keywords": [
            "result", "merit", "merit list", "rank", "air", "all india rank",
            "scorecard", "score card", "percentile", "cut off", "cutoff",
            "qualifying marks", "minimum marks", "tie breaking", "tie-breaking"
        ]
    }
}

# Categories for State Counselling Brochure chunks
# Categories for State Counselling Brochure chunks
# NOTE: These categories are for metadata/display purposes only - NOT used for retrieval filtering.
STATE_BROCHURE_CATEGORIES = {
    "seat_matrix": {
        "description": "College wise seats, total seats, seat distribution",
        "keywords": [
            "seat matrix", "seats", "total seats", "mbbs seats", "bds seats",
            "college wise", "institute wise", "intake", "sanctioned seats",
            "government seats", "private seats", "management seats"
        ]
    },
    "state_reservation": {
        "description": "State specific reservation and communities",
        "keywords": [
            "state reservation", "reservation policy", "85%", "state obc", "state sc",
            "state st", "local reservation", "domicile reservation",
            "community reservation", "caste", "backward class", "most backward",
            "tea garden", "tea tribe", "plains tribe", "hill tribe",
            "kannada medium", "rural quota", "hyderabad karnataka"
        ]
    },
    "counselling_process": {
        "description": "State counselling rounds and procedures",
        "keywords": [
            "counselling round", "round 1", "round 2", "mop up", "mop-up",
            "stray vacancy", "document verification", "reporting",
            "choice filling", "option entry", "allotment", "upgrade",
            "seat acceptance", "willingness", "exit option", "surrender",
            "counselling schedule", "important dates"
        ]
    },
    "eligibility": {
        "description": "State specific eligibility and domicile rules",
        "keywords": [
            "domicile", "nativity", "state eligibility", "resident",
            "residence", "years of study", "studied in state",
            "parent domicile", "state candidate", "local candidate"
        ]
    },
    "fee_structure": {
        "description": "Tuition fee, NRI fee, management fee, college-wise fee tables",
        "keywords": [
            "fee structure", "tuition fee", "nri fee", "nri quota fee",
            "management fee", "government college fee", "private college fee",
            "hostel fee", "annual fee", "semester fee",
            "government medical college", "medical college", "admission fee",
            "university charges", "security fee", "refundable", "mbbs 1st year",
            "net amount", "institutional", "pg diploma", "aiims"
        ]
    }
}


# ============== KEYWORD-BASED CLASSIFIER ==============

def classify_by_keywords(text: str, document_type: str) -> Optional[str]:
    """
    Try to classify chunk using keyword matching (no LLM cost).
    This is for metadata/display purposes only - NOT used for retrieval filtering.
    Returns category if confident match found, None otherwise.
    """
    text_lower = text.lower()
    
    # Select category set based on document type
    if document_type == "nta_bulletin":
        categories = NTA_BULLETIN_CATEGORIES
    elif document_type in ("state_counseling", "college_info", "cutoffs", "other"):
        # college_info = fee/college PDFs — share state-brochure keyword buckets (fees, seats, …)
        categories = STATE_BROCHURE_CATEGORIES
    else:
        return None
    
    # Count keyword matches per category
    scores = {}
    for category, config in categories.items():
        score = sum(1 for kw in config["keywords"] if kw in text_lower)
        if score > 0:
            scores[category] = score
    
    if not scores:
        return None  # No keywords matched, need LLM
    
    # Get category with highest score
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    
    # Require at least 2 keyword matches for confidence
    if best_score >= 2:
        return best_category
    
    return None  # Not confident enough, need LLM


# ============== LLM-BASED CLASSIFIER ==============

def classify_by_llm(text: str, document_type: str) -> Dict[str, any]:
    """
    Classify chunk using GPT-4o-mini when keywords are not sufficient
    Returns: {"category": str, "section": str, "importance": str}
    """
    
    if document_type == "nta_bulletin":
        categories_desc = """
- exam_info: Exam syllabus, pattern, marking scheme, dates, admit card, exam centers
- eligibility: Age limit, educational qualifications, attempt limits
- application: Application fee, form filling, required documents for application
- counselling: MCC, AIQ, AIIMS, JIPMER, central quota counselling process
- reservation: Central level reservation (OBC 27%, SC 15%, ST 7.5%, EWS 10%, PwBD)
- result: Merit list, qualifying marks, rank, scorecard, tie-breaking rules
- general: Anything that doesn't fit above categories
"""
    else:
        categories_desc = """
- seat_matrix: College wise seats, total seats, seat distribution
- state_reservation: State specific reservation categories and communities
- counselling_process: State counselling rounds, document verification, allotment
- eligibility: State domicile rules, nativity requirements
- fee_structure: Tuition fee, college-wise fees, hostel, NRI quota fee, MBBS year-wise fees
- general: Anything that doesn't fit above categories
"""

    doc_label = "NTA Bulletin"
    if document_type == "nta_bulletin":
        doc_label = "NTA Bulletin"
    elif document_type == "college_info":
        doc_label = "State College / Fee Information Document"
    else:
        doc_label = "State Counselling Brochure"

    prompt = f"""Classify this document chunk from a NEET {doc_label}.

CHUNK TEXT:
{text[:2000]}

CATEGORIES:
{categories_desc}

Respond in exactly this format (no extra text):
CATEGORY: <category_name>
SECTION: <2-3 word topic>
IMPORTANCE: <high/medium/low>

Rules for importance:
- high: Key eligibility criteria, important dates, fee amounts, seat numbers
- medium: Process descriptions, general rules
- low: Examples, clarifications, minor details
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse response
        category = "general"
        section = "General Information"
        importance = "medium"
        
        for line in result_text.split("\n"):
            if line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip().lower()
            elif line.startswith("SECTION:"):
                section = line.replace("SECTION:", "").strip()
            elif line.startswith("IMPORTANCE:"):
                importance = line.replace("IMPORTANCE:", "").strip().lower()
        
        return {
            "category": category,
            "section": section,
            "importance": importance
        }
        
    except Exception as e:
        print(f"LLM classification error: {e}")
        return {
            "category": "general",
            "section": "General Information",
            "importance": "medium"
        }


# ============== MAIN CLASSIFIER FUNCTION ==============

def classify_chunk(text: str, document_type: str, state: str = "All-India") -> Dict[str, any]:
    """
    Main function to classify a chunk before indexing to the vector store.
    
    Args:
        text: The chunk text to classify
        document_type: "nta_bulletin" or "state_counseling"
        state: The state for state brochures, "All-India" for NTA bulletin
    
    Returns:
        Dictionary with classification metadata:
        {
            "category": str,
            "section": str,
            "importance": str,
            "document_type": str,
            "state": str
        }
    """
    
    # Try keyword-based classification first (free)
    category = classify_by_keywords(text, document_type)
    
    if category:
        # Keywords were sufficient
        return {
            "category": category,
            "section": get_section_from_category(category),
            "importance": estimate_importance(text),
            "document_type": document_type,
            "state": state
        }
    else:
        # Need LLM classification
        llm_result = classify_by_llm(text, document_type)
        llm_result["document_type"] = document_type
        llm_result["state"] = state
        return llm_result


def get_section_from_category(category: str) -> str:
    """Generate section name from category"""
    section_map = {
        "exam_info": "Exam Information",
        "eligibility": "Eligibility Criteria",
        "application": "Application Process",
        "counselling": "Counselling Process",
        "reservation": "Reservation Policy",
        "result": "Result & Merit",
        "seat_matrix": "Seat Distribution",
        "state_reservation": "State Reservation",
        "counselling_process": "Counselling Rounds",
        "fee_structure": "Fee Structure",
        "general": "General Information"
    }
    return section_map.get(category, "General Information")


def estimate_importance(text: str) -> str:
    """Estimate importance based on content patterns"""
    text_lower = text.lower()
    
    # High importance indicators
    high_indicators = [
        "last date", "deadline", "important date", "must", "mandatory",
        "eligibility criteria", "required", "fee of rs", "total seats",
        "cutoff", "qualifying marks", "minimum marks"
    ]
    
    # Low importance indicators  
    low_indicators = [
        "for example", "e.g.", "such as", "note:", "clarification",
        "illustration", "query", "faq"
    ]
    
    if any(ind in text_lower for ind in high_indicators):
        return "high"
    elif any(ind in text_lower for ind in low_indicators):
        return "low"
    else:
        return "medium"


# ============== BATCH RECLASSIFICATION ==============

async def reclassify_existing_chunks(vector_index, batch_size: int = 100):
    """
    Reclassify existing chunks that have category='general'
    This is for migrating old indexed documents
    """
    # This would need to:
    # 1. Query vector store for chunks with category='general'
    # 2. For each chunk, run classify_chunk()
    # 3. Update the metadata in vector store
    # 
    # Implementation depends on how you want to batch this
    pass
