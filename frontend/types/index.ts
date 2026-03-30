// Model types
export type ModelType = 'openai' | 'huggingface';

// User roles
export type UserRole = 'student' | 'admin' | 'super_admin';

// User type
export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  phone: string | null;
  age: number | null;
  role: UserRole;
  is_active: boolean;
  is_verified: boolean;
  target_exams: string[];
  preferences?: {
    preferred_state?: string;
    category?: string;
    target_colleges?: string[];
  };
  created_at: string;
}

// Auth response
export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

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
  needsClarification?: boolean;
  clarificationOptions?: string[];
  originalQuestion?: string;
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
