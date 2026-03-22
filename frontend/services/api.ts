import { ChatResponse, ModelType, Source } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Send a chat message to the RAG backend
 */
export async function sendChatMessage(
  question: string,
  model: ModelType
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      model,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export type StreamDonePayload = {
  sources?: Source[];
  model_used: string;
};

/**
 * Stream chat from /chat/stream (NDJSON: delta lines, then done).
 */
export async function streamChatMessage(
  question: string,
  model: ModelType,
  onDelta: (text: string) => void,
  onDone: (data: StreamDonePayload) => void
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, model }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(
      typeof error.detail === 'string'
        ? error.detail
        : `HTTP error! status: ${response.status}`
    );
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
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let obj: { type: string; text?: string; message?: string; sources?: Source[]; model_used?: string };
      try {
        obj = JSON.parse(trimmed);
      } catch {
        continue;
      }
      if (obj.type === 'delta' && obj.text) {
        onDelta(obj.text);
      } else if (obj.type === 'done') {
        onDone({
          sources: obj.sources,
          model_used: obj.model_used ?? 'openai',
        });
      } else if (obj.type === 'error') {
        throw new Error(obj.message || 'Stream error');
      }
    }
  }

  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim());
      if (obj.type === 'done') {
        onDone({
          sources: obj.sources,
          model_used: obj.model_used ?? 'openai',
        });
      } else if (obj.type === 'error') {
        throw new Error(obj.message || 'Stream error');
      }
    } catch (e) {
      if (e instanceof SyntaxError) return;
      throw e;
    }
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
