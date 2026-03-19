# RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot that allows users to chat with their PDF documents using AI.

## Features

- 🔍 **RAG-based Q&A** - Ask questions and get answers based only on your documents
- 🤖 **Multi-LLM Support** - Choose between OpenAI GPT-4o-mini or HuggingFace Mistral-7B
- 💬 **ChatGPT-like UI** - Modern, responsive chat interface
- 📄 **PDF Support** - Upload and query PDF documents
- 🔗 **Source Citations** - See which parts of documents were used to generate answers
- ⚡ **Fast Retrieval** - FAISS vector database for efficient similarity search

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Chat UI    │  │   Model     │  │      API Service        │  │
│  │  Component  │  │  Selector   │  │   (services/api.ts)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────────┐
│                         Backend (FastAPI)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  /chat      │  │   RAG       │  │    LLM Providers        │  │
│  │  endpoint   │──│   Query     │──│  (OpenAI, HuggingFace)  │  │
│  └─────────────┘  │   Engine    │  └─────────────────────────┘  │
│                   └──────┬──────┘                                │
│                          │                                       │
│  ┌───────────────────────▼──────────────────────────────────┐   │
│  │                    LlamaIndex                             │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │   │
│  │  │  FAISS   │  │  Embeddings  │  │   Document Loader  │  │   │
│  │  │  Vector  │  │  (MiniLM)    │  │   (PDF Reader)     │  │   │
│  │  │  Store   │  └──────────────┘  └────────────────────┘  │   │
│  │  └──────────┘                                             │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LlamaIndex** - RAG framework
- **FAISS** - Vector database
- **sentence-transformers** - Embeddings (all-MiniLM-L6-v2)

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── ingest.py           # Document ingestion
│   ├── query.py            # RAG query engine
│   ├── requirements.txt    # Python dependencies
│   ├── data/               # PDF documents folder
│   ├── storage/            # Persisted FAISS index
│   └── llm_providers/
│       ├── __init__.py
│       ├── factory.py      # LLM factory
│       ├── openai_provider.py
│       └── hf_provider.py
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx        # Main chat page
    │   └── globals.css
    ├── components/
    │   ├── ChatWindow.tsx
    │   ├── MessageBubble.tsx
    │   └── ModelSelector.tsx
    ├── services/
    │   └── api.ts          # API client
    ├── types/
    │   └── index.ts        # TypeScript types
    └── package.json
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Add Your API Keys

Edit `backend/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### 4. Add PDF Documents

Place your PDF files in `backend/data/` folder.

### 5. Run Document Ingestion

```bash
# From backend folder
python ingest.py
```

This will:
- Load PDFs from `data/` folder
- Chunk documents (800 tokens, 150 overlap)
- Create embeddings using MiniLM
- Store in FAISS vector database
- Persist index to `storage/` folder

### 6. Start Backend Server

```bash
# From backend folder
uvicorn app:app --reload --port 8000
```

API will be available at `http://localhost:8000`

### 7. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file (optional)
cp .env.example .env.local

# Start development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| POST | `/chat` | Send chat message |
| GET | `/models` | Get available models |
| POST | `/ingest` | Trigger re-ingestion |

### Chat Request

```json
POST /chat
{
  "question": "What is this document about?",
  "model": "openai"  // or "huggingface"
}
```

### Chat Response

```json
{
  "answer": "This document discusses...",
  "sources": [
    {
      "file_name": "document.pdf",
      "page": 5,
      "text_snippet": "..."
    }
  ],
  "model_used": "openai"
}
```

## Configuration

### RAG Settings (in `ingest.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| CHUNK_SIZE | 800 | Tokens per chunk |
| CHUNK_OVERLAP | 150 | Overlap between chunks |
| EMBED_MODEL | all-MiniLM-L6-v2 | Sentence transformer model |
| TOP_K | 3 | Number of chunks to retrieve |

### Supported Models

| Model | Provider | Description |
|-------|----------|-------------|
| GPT-4o-mini | OpenAI | Fast, efficient GPT-4 variant |
| Mistral-7B-Instruct | HuggingFace | Open-source instruction model |

## Deployment

### Backend (Render)

1. Create a new Web Service on Render
2. Connect your repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (API keys)

### Frontend (Vercel)

1. Import project to Vercel
2. Set framework preset to Next.js
3. Add environment variable: `NEXT_PUBLIC_API_URL=<your-backend-url>`
4. Deploy

## Development

### Running Tests

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Adding New LLM Providers

1. Create new provider file in `backend/llm_providers/`
2. Implement `get_<provider>_llm()` function
3. Register in `factory.py`

## Troubleshooting

### "No documents found"
- Ensure PDFs are in `backend/data/` folder
- Run `python ingest.py` to index documents

### "API key not set"
- Check `.env` file exists in backend folder
- Verify API keys are correct

### CORS errors
- Check backend CORS settings in `app.py`
- Ensure frontend URL is in allowed origins

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
