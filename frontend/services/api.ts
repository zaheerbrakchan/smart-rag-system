import { ChatResponse, ModelType } from '@/types';

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
