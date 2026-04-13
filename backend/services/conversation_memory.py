"""
Conversation Memory Service
Uses LlamaIndex ChatMemoryBuffer for conversation context management
"""

import os
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole as LlamaMessageRole
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from models.conversation import Conversation, Message, MessageRole


# Configuration
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "2000"))  # Max tokens for memory
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "10"))  # Max messages to load


# State detection for context extraction
STATES_FOR_CONTEXT = {
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "andhra pradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "telangana": "Telangana",
    "maharashtra": "Maharashtra",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "madhya pradesh": "Madhya Pradesh",
    "uttar pradesh": "Uttar Pradesh",
    "bihar": "Bihar",
    "west bengal": "West Bengal",
    "odisha": "Odisha",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "delhi": "Delhi",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "uttarakhand": "Uttarakhand",
    "himachal pradesh": "Himachal Pradesh",
    "goa": "Goa",
    "jammu and kashmir": "Jammu & Kashmir",
    "jammu": "Jammu & Kashmir",
    "kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "jk": "Jammu & Kashmir",
    "puducherry": "Puducherry",
    "assam": "Assam",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura",
    "arunachal pradesh": "Arunachal Pradesh",
    "chandigarh": "Chandigarh",
    "ladakh": "Ladakh",
}


class ConversationMemory:
    """
    Manages conversation memory using LlamaIndex ChatMemoryBuffer.
    
    - Loads history from PostgreSQL Message table
    - Maintains token-limited context window
    - Provides formatted history for RAG prompts
    - Extracts context (state, topic) for query routing
    """
    
    def __init__(self, conversation_id: Optional[int] = None, user_id: Optional[int] = None):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)
        self._raw_messages: List[Message] = []  # Store raw DB messages for metadata access
    
    async def load_from_db(self, db: AsyncSession) -> None:
        """Load conversation history from database into memory buffer"""
        if not self.conversation_id:
            return
        
        # Fetch recent messages (limited by MEMORY_TOP_K)
        query = (
            select(Message)
            .where(Message.conversation_id == self.conversation_id)
            .order_by(desc(Message.created_at))
            .limit(MEMORY_TOP_K)
        )
        result = await db.execute(query)
        messages = result.scalars().all()
        
        # Reverse to get chronological order (oldest first)
        messages = list(reversed(messages))
        self._raw_messages = messages
        
        # Add to LlamaIndex memory buffer
        for msg in messages:
            role = self._map_role(msg.role)
            chat_msg = ChatMessage(role=role, content=msg.content)
            self.memory.put(chat_msg)
    
    def _map_role(self, role: MessageRole) -> LlamaMessageRole:
        """Map database MessageRole to LlamaIndex MessageRole"""
        mapping = {
            MessageRole.USER: LlamaMessageRole.USER,
            MessageRole.ASSISTANT: LlamaMessageRole.ASSISTANT,
            MessageRole.SYSTEM: LlamaMessageRole.SYSTEM,
        }
        return mapping.get(role, LlamaMessageRole.USER)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the memory buffer"""
        llama_role = LlamaMessageRole.USER if role == "user" else LlamaMessageRole.ASSISTANT
        self.memory.put(ChatMessage(role=llama_role, content=content))
    
    def get_chat_history(self) -> List[ChatMessage]:
        """Get all messages in the memory buffer"""
        return self.memory.get_all()
    
    def get_formatted_history(self, max_messages: Optional[int] = None) -> str:
        """
        Get conversation history formatted for inclusion in RAG prompt.
        Returns empty string if no history.
        """
        messages = self.memory.get_all()
        if not messages:
            return ""
        
        if max_messages:
            messages = messages[-max_messages:]
        
        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == LlamaMessageRole.USER else "Assistant"
            # Truncate long messages for context efficiency
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            formatted_lines.append(f"{role_label}: {content}")
        
        return "\n".join(formatted_lines)
    
    def extract_conversation_context(self) -> Dict[str, Any]:
        """
        Extract context from conversation history for intelligent routing.
        
        Returns:
            dict with keys:
            - detected_state: State mentioned in conversation (most recent)
            - detected_topic: Topic being discussed (fees, eligibility, etc.)
            - last_user_question: The most recent user question for context
            - is_followup: Whether current context suggests a follow-up question
        """
        context = {
            "detected_state": None,
            "detected_topic": None,
            "last_user_question": None,
            "is_followup": False,
        }
        
        messages = self.memory.get_all()
        if not messages:
            return context
        
        # Scan messages in reverse (most recent first) to find state
        all_text = ""
        for msg in reversed(messages):
            text_lower = msg.content.lower()
            all_text += " " + text_lower
            
            # Detect state (prioritize most recent mention)
            if not context["detected_state"]:
                for key, value in STATES_FOR_CONTEXT.items():
                    pattern = r'\b' + re.escape(key) + r'\b'
                    if re.search(pattern, text_lower):
                        context["detected_state"] = value
                        break
            
            # Get last user question
            if msg.role == LlamaMessageRole.USER and not context["last_user_question"]:
                context["last_user_question"] = msg.content
        
        # Detect topic from conversation
        topic_keywords = {
            "fee": ["fee", "fees", "cost", "payment", "amount", "price", "charges", "tuition"],
            "eligibility": ["eligible", "eligibility", "qualify", "criteria", "requirement", "age limit"],
            "dates": ["date", "deadline", "when", "schedule", "last date"],
            "colleges": ["college", "colleges", "seat", "seats", "institute", "university"],
            "cutoff": ["cutoff", "cut off", "rank", "score", "marks", "percentile"],
            "reservation": ["reservation", "quota", "obc", "sc", "st", "ews", "category", "reserved"],
            "process": ["process", "procedure", "how to", "steps", "apply", "counselling"],
            "documents": ["document", "documents", "certificate", "required documents"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text for kw in keywords):
                context["detected_topic"] = topic
                break
        
        # Check if this looks like a follow-up
        context["is_followup"] = len(messages) > 0
        
        return context
    
    def is_vague_followup(self, query: str) -> bool:
        """
        Check if a query is a vague follow-up that needs reframing.
        
        Examples of vague follow-ups:
        - "what about ST category?"
        - "for OBC?"
        - "and the fees?"
        - "what is it for general?"
        """
        query_lower = query.lower().strip()
        
        # Short queries are likely follow-ups
        word_count = len(query_lower.split())
        if word_count <= 6:
            # Check for follow-up indicators
            followup_patterns = [
                r'^(what|how)\s+(about|for|is)',  # "what about...", "what for..."
                r'^(and|or)\s+(the|for|what)',     # "and the...", "and for..."
                r'^for\s+\w+\??$',                  # "for OBC?", "for ST?"
                r'^(what|how)\s+is\s+(it|this|that)',  # "what is it for..."
                r'category\s*\??$',                 # ends with "category?"
                r'^same\s+for',                     # "same for..."
                r'^(tell|show)\s+me\s+for',        # "tell me for..."
            ]
            
            for pattern in followup_patterns:
                if re.search(pattern, query_lower):
                    return True
            
            # If we have conversation history and query is short, likely a follow-up
            if self.memory.get_all() and word_count <= 5:
                return True
        
        return False
    
    def reframe_query_with_context(self, query: str) -> str:
        """
        Reframe a vague follow-up question using conversation context.
        
        Example:
        - Previous: "What is the NEET fee for J&K?"
        - Current: "what about ST category?"
        - Reframed: "What is the NEET fee for ST category in Jammu & Kashmir?"
        """
        if not self.is_vague_followup(query):
            return query
        
        ctx = self.extract_conversation_context()
        
        # Must have context to reframe
        if not ctx["detected_state"] and not ctx["detected_topic"] and not ctx["last_user_question"]:
            return query
        
        # Build reframed query parts
        parts = []
        
        # Extract the new aspect from current query
        query_lower = query.lower()
        
        # Detect category mentions
        categories = {
            "st": "ST (Scheduled Tribe) category",
            "sc": "SC (Scheduled Caste) category", 
            "obc": "OBC category",
            "ews": "EWS category",
            "general": "General category",
            "pwd": "PwD category",
            "ur": "Unreserved/General category",
        }
        
        new_category = None
        for key, value in categories.items():
            if re.search(r'\b' + key + r'\b', query_lower):
                new_category = value
                break
        
        # Get topic from context
        topic = ctx.get("detected_topic", "")
        state = ctx.get("detected_state", "")
        prev_question = ctx.get("last_user_question", "")
        
        # Build reframed query
        if new_category and state:
            if topic == "fee":
                reframed = f"What is the NEET counselling fee structure for {new_category} in {state}?"
            elif topic == "reservation":
                reframed = f"What is the reservation policy for {new_category} in {state} NEET counselling?"
            elif topic == "cutoff":
                reframed = f"What is the NEET cutoff for {new_category} in {state}?"
            elif topic == "eligibility":
                reframed = f"What is the eligibility criteria for {new_category} in {state} NEET counselling?"
            elif topic == "colleges":
                reframed = f"Which colleges are available for {new_category} in {state}?"
            else:
                # Generic reframe using previous question as template
                reframed = f"{prev_question.rstrip('?')} for {new_category}?"
        elif state and not new_category:
            # Just has state, append to query
            if state.lower() not in query_lower:
                reframed = f"{query.rstrip('?')} in {state}?"
            else:
                reframed = query
        else:
            # Can't meaningfully reframe
            reframed = query
        
        return reframed
    
    def get_routing_context_prompt(self) -> str:
        """
        Get a concise context string for the query router.
        Helps router understand follow-up questions.
        """
        ctx = self.extract_conversation_context()
        
        parts = []
        if ctx["detected_state"]:
            parts.append(f"Current state context: {ctx['detected_state']}")
        if ctx["detected_topic"]:
            parts.append(f"Topic: {ctx['detected_topic']}")
        if ctx["last_user_question"]:
            parts.append(f"Previous question: {ctx['last_user_question'][:150]}")
        
        if parts:
            return "CONVERSATION CONTEXT:\n" + "\n".join(parts)
        return ""
    
    def get_context_summary(self) -> Optional[str]:
        """
        Generate a brief summary of conversation context.
        Useful for understanding user's ongoing needs.
        """
        messages = self.memory.get_all()
        if len(messages) < 2:
            return None
        
        # Extract key topics from recent user messages
        user_messages = [m.content for m in messages if m.role == LlamaMessageRole.USER]
        if not user_messages:
            return None
        
        # Return last few user queries as context hint
        recent = user_messages[-3:]
        return "Recent questions: " + " | ".join([q[:100] for q in recent])
    
    def clear(self) -> None:
        """Clear the memory buffer"""
        self.memory.reset()


async def get_or_create_conversation(
    db: AsyncSession,
    user_id: int,
    conversation_id: Optional[int] = None
) -> Conversation:
    """
    Get existing conversation or create new one.
    Returns the conversation object.
    """
    if conversation_id:
        conversation = await db.get(Conversation, conversation_id)
        if conversation and conversation.user_id == user_id:
            return conversation
    
    # Create new conversation
    conversation = Conversation(user_id=user_id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def save_message_to_db(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    model_used: Optional[str] = None,
    was_faq_match: bool = False,
    faq_confidence: Optional[float] = None,
    filters_applied: Optional[Dict] = None,
    response_time_ms: Optional[int] = None
) -> Message:
    """Save a message to the database"""
    msg_role = MessageRole.USER if role == "user" else MessageRole.ASSISTANT
    
    message = Message(
        conversation_id=conversation_id,
        role=msg_role,
        content=content,
        sources=sources or [],
        model_used=model_used,
        was_faq_match=was_faq_match,
        faq_confidence=faq_confidence,
        filters_applied=filters_applied or {},
        response_time_ms=response_time_ms
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def generate_conversation_title(question: str) -> str:
    """
    Generate a concise, meaningful title for a conversation based on the first question.
    
    Args:
        question: The first user question in the conversation
    
    Returns:
        A short (3-6 word) descriptive title
    """
    from openai import OpenAI
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Generate a very short title (3-6 words) for this NEET counselling conversation based on the user's question.
Rules:
- Maximum 6 words
- No punctuation at the end
- Capture the main topic/intent
- Be specific (e.g., "J&K Counselling Fees" not "Fee Question")
- For states, use abbreviations if long (e.g., "MP" for Madhya Pradesh)

Examples:
- "What is fee in Karnataka?" → "Karnataka Counselling Fees"
- "eligibility for neet 2026" → "NEET 2026 Eligibility Criteria"
- "What is the reservation for OBC in Bihar?" → "Bihar OBC Reservation Policy"
- "important dates for counselling" → "Counselling Important Dates"
- "documents required for registration" → "NEET Registration Documents"""
                },
                {
                    "role": "user",
                    "content": f"Question: {question[:200]}"
                }
            ],
            temperature=0.3,
            max_tokens=20
        )
        
        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"\'')
        # Limit length
        if len(title) > 50:
            title = title[:47] + "..."
        return title
        
    except Exception as e:
        # Fallback: extract key words from question
        import re
        # Remove common words and create simple title
        words = question.lower().split()[:6]
        stop_words = {'what', 'is', 'the', 'for', 'in', 'a', 'an', 'how', 'can', 'i', 'do', 'does', 'are', 'to', 'of'}
        key_words = [w.capitalize() for w in words if w not in stop_words][:4]
        if key_words:
            return ' '.join(key_words)
        return f"Query: {question[:30]}..."


async def update_conversation_title(
    db: AsyncSession,
    conversation_id: int,
    title: str
) -> None:
    """Update the title of a conversation."""
    conversation = await db.get(Conversation, conversation_id)
    if conversation:
        conversation.title = title
        await db.commit()


def build_prompt_with_memory(
    base_prompt: str,
    memory: ConversationMemory,
    include_history: bool = True,
    max_history_messages: int = 5
) -> str:
    """
    Enhance a RAG prompt with conversation history context.
    
    Args:
        base_prompt: The original RAG prompt
        memory: ConversationMemory instance with loaded history
        include_history: Whether to include conversation history
        max_history_messages: Max number of history messages to include
    
    Returns:
        Enhanced prompt with conversation context
    """
    if not include_history:
        return base_prompt
    
    history = memory.get_formatted_history(max_messages=max_history_messages)
    if not history:
        return base_prompt
    
    # Insert conversation context before the question
    context_section = f"""
CONVERSATION HISTORY (for context - answer the current question):
{history}

---

"""
    
    # Find where to insert (before "Question:" or at the end)
    if "Question:" in base_prompt:
        parts = base_prompt.rsplit("Question:", 1)
        return parts[0] + context_section + "Question:" + parts[1]
    else:
        return base_prompt + "\n" + context_section
