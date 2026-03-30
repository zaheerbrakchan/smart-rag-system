import { ChatResponse, ModelType } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// User preferences for smart query routing
export interface UserPreferences {
  preferred_state?: string;
  category?: string;
  target_exams?: string[];
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
 */
export async function streamChatMessage(
  question: string,
  model: ModelType,
  onToken: (token: string) => void,
  onSources: (sources: any[]) => void,
  onDone: (filters?: any) => void,
  onError: (error: string) => void,
  userPreferences?: UserPreferences,
  clarifiedScope?: string,
  onClarificationNeeded?: (options: string[], message: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        model,
        user_preferences: userPreferences,
        clarified_scope: clarifiedScope,
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
            } else if (data.type === 'done') {
              onDone(data.filters_applied);
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
