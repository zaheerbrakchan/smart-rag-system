// Model types
export type ModelType = 'openai' | 'huggingface';

// Source reference from RAG
export interface Source {
  file_name: string;
  page?: number;
  text_snippet: string;
}

// Chat message
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
  modelUsed?: string;
  isError?: boolean;
}

// API response
export interface ChatResponse {
  answer: string;
  sources?: Source[];
  model_used: string;
}

// Model info
export interface ModelInfo {
  id: ModelType;
  name: string;
  provider: string;
  description: string;
}
