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

export type SupportQueryStatus = 'pending' | 'in_progress' | 'answered' | 'closed';

export interface SupportQueryReply {
  id: number;
  responder_admin_id: number | null;
  reply_text: string;
  sent_email: boolean;
  sent_sms: boolean;
  created_at: string;
}

export interface SupportQuery {
  id: number;
  user_id: number;
  student_name: string;
  phone: string;
  email: string | null;
  subject: string;
  message: string;
  status: SupportQueryStatus;
  assigned_admin_id: number | null;
  answered_at: string | null;
  created_at: string;
  updated_at: string;
  replies: SupportQueryReply[];
}

export interface SupportNotification {
  id: number;
  type: string;
  title: string;
  body: string;
  related_query_id: number | null;
  is_read: boolean;
  created_at: string;
}

function extractApiErrorMessage(error: any, fallback: string): string {
  const detail = error?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const firstMsg = detail.find((d) => typeof d?.msg === 'string')?.msg;
    if (firstMsg) return firstMsg;
    return detail.map((d) => d?.msg || d?.message || '').filter(Boolean).join(', ') || fallback;
  }
  if (typeof error?.message === 'string' && error.message.trim()) return error.message;
  return fallback;
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
  onTitle?: (title: string, conversationId: number) => void,
  preferredLanguage?: 'en' | 'hi' | 'mr'
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
        preferred_language: preferredLanguage,
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

export async function createSupportQuery(
  token: string,
  payload: {
    student_name?: string;
    phone?: string;
    email?: string;
    subject?: string;
    message: string;
  }
): Promise<SupportQuery> {
  const response = await fetch(`${API_BASE_URL}/support/queries`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to submit support query' }));
    throw new Error(extractApiErrorMessage(error, 'Failed to submit support query'));
  }
  return response.json();
}

export async function getMySupportQueries(token: string): Promise<SupportQuery[]> {
  const response = await fetch(`${API_BASE_URL}/support/queries/me`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error('Failed to fetch support queries');
  }
  return response.json();
}

export async function getMySupportNotifications(token: string): Promise<SupportNotification[]> {
  const response = await fetch(`${API_BASE_URL}/support/notifications/me`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error('Failed to fetch notifications');
  }
  return response.json();
}

export async function markSupportNotificationRead(token: string, notificationId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/support/notifications/${notificationId}/read`, {
    method: 'PATCH',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error('Failed to mark notification as read');
  }
}

export interface AdminSupportQueryList {
  queries: SupportQuery[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export async function getAdminSupportQueries(
  token: string,
  params: { page?: number; page_size?: number; status?: SupportQueryStatus | ''; search?: string } = {}
): Promise<AdminSupportQueryList> {
  const qs = new URLSearchParams();
  if (params.page) qs.set('page', String(params.page));
  if (params.page_size) qs.set('page_size', String(params.page_size));
  if (params.status) qs.set('status', params.status);
  if (params.search) qs.set('search', params.search);
  const response = await fetch(`${API_BASE_URL}/admin/support/queries?${qs.toString()}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error('Failed to fetch admin support queries');
  }
  return response.json();
}

export async function updateAdminSupportQueryStatus(
  token: string,
  queryId: number,
  payload: { status: SupportQueryStatus; assigned_admin_id?: number | null }
): Promise<SupportQuery> {
  const response = await fetch(`${API_BASE_URL}/admin/support/queries/${queryId}`, {
    method: 'PATCH',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to update support query' }));
    throw new Error(error.detail || 'Failed to update support query');
  }
  return response.json();
}

export async function replyAdminSupportQuery(
  token: string,
  queryId: number,
  replyText: string
): Promise<SupportQuery> {
  const response = await fetch(`${API_BASE_URL}/admin/support/queries/${queryId}/reply`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ reply_text: replyText }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to send reply' }));
    throw new Error(error.detail || 'Failed to send reply');
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
