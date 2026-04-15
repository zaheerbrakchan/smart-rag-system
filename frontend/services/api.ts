import { ChatResponse, ModelType } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Use V2 unified endpoint (tool-based architecture)
// Set to true to use the new single-prompt architecture
const USE_V2_ENDPOINT = process.env.NEXT_PUBLIC_USE_V2_CHAT === 'true' || true;

// User preferences for smart query routing
export interface UserPreferences {
  preferred_state?: string;
  category?: string;
}

/**
 * Send a chat message to the RAG backend (non-streaming)
 */
export async function sendChatMessage(
  question: string,
  model: ModelType,
  userPreferences?: UserPreferences
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      model,
      user_preferences: userPreferences,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Stream chat response from the RAG backend
 * Uses V2 unified endpoint by default (single-prompt with tool calling)
 */
export async function streamChatMessage(
  question: string,
  model: ModelType,
  onToken: (token: string) => void,
  onSources: (sources: any[]) => void,
  onDone: (filters?: any, conversationId?: number) => void,
  onError: (error: string) => void,
  userPreferences?: UserPreferences,
  clarifiedScope?: string,
  onClarificationNeeded?: (options: string[], message: string) => void,
  onSuggestedReplies?: (replies: string[]) => void,
  conversationId?: number,
  userId?: number,
  onTitle?: (title: string, conversationId: number) => void
): Promise<void> {
  try {
    // Select endpoint based on config
    const endpoint = USE_V2_ENDPOINT ? '/chat/v2/stream' : '/chat/stream';
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        model,
        user_preferences: userPreferences,
        clarified_scope: clarifiedScope,
        conversation_id: conversationId,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'token' && data.token) {
              onToken(data.token);
            } else if (data.type === 'sources' && data.sources) {
              onSources(data.sources);
            } else if (data.type === 'clarification_needed' && onClarificationNeeded) {
              onClarificationNeeded(data.options, data.message);
            } else if (data.type === 'suggested_replies' && onSuggestedReplies) {
              onSuggestedReplies(data.replies || []);
            } else if (data.type === 'done') {
              onDone(data.filters_applied, data.conversation_id);
            } else if (data.type === 'title' && onTitle) {
              onTitle(data.title, data.conversation_id);
            } else if (data.type === 'error') {
              onError(data.error);
            }
          } catch (e) {
            // Ignore parse errors for incomplete JSON
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Stream failed');
  }
}

/**
 * Check API health status
 */
export async function checkHealth(): Promise<{
  status: string;
  index_loaded: boolean;
  available_models: string[];
}> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error('API health check failed');
  }

  return response.json();
}

/**
 * Get available models
 */
export async function getModels(): Promise<{
  models: Array<{
    id: ModelType;
    name: string;
    provider: string;
    description: string;
  }>;
}> {
  const response = await fetch(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }

  return response.json();
}

/**
 * Trigger document re-ingestion
 */
export async function triggerIngestion(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Ingestion failed' }));
    throw new Error(error.detail);
  }

  return response.json();
}

// ============== CONVERSATION APIs ==============

export interface ConversationSummary {
  id: number;
  title: string | null;
  summary: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources: any[] | null;
  filters_applied: any | null;
  was_faq_match: boolean;
  created_at: string;
}

export interface ConversationDetail {
  id: number;
  title: string | null;
  summary: string | null;
  messages: ConversationMessage[];
  created_at: string;
  updated_at: string;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  total: number;
  page: number;
  page_size: number;
}

/**
 * Get list of user's conversations
 */
export async function getConversations(
  token: string,
  page: number = 1,
  pageSize: number = 20
): Promise<ConversationListResponse> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/?page=${page}&page_size=${pageSize}`,
    {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to fetch conversations');
  }

  return response.json();
}

/**
 * Get a specific conversation with all messages
 */
export async function getConversation(
  token: string,
  conversationId: number
): Promise<ConversationDetail> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}`,
    {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to fetch conversation');
  }

  return response.json();
}

/**
 * Create a new conversation
 */
export async function createConversation(
  token: string,
  title?: string
): Promise<ConversationSummary> {
  const response = await fetch(`${API_BASE_URL}/conversations/`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error('Failed to create conversation');
  }

  return response.json();
}

/**
 * Update conversation title
 */
export async function updateConversation(
  token: string,
  conversationId: number,
  title: string
): Promise<ConversationSummary> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}`,
    {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title }),
    }
  );

  if (!response.ok) {
    throw new Error('Failed to update conversation');
  }

  return response.json();
}

/**
 * Delete a conversation
 */
export async function deleteConversation(
  token: string,
  conversationId: number
): Promise<{ success: boolean; deleted_id: number }> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}`,
    {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    }
  );

  if (!response.ok) {
    throw new Error('Failed to delete conversation');
  }

  return response.json();
}
