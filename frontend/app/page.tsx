'use client';

import { useState, useRef, useEffect } from 'react';
import ChatWindow from '@/components/ChatWindow';
import { Message, ModelType } from '@/types';
import { streamChatMessage } from '@/services/api';
import { GraduationCap, Send, Sparkles, BookOpen, Calendar, FileCheck, HelpCircle, Shield, RotateCcw } from 'lucide-react';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedModel, setSelectedModel] = useState<ModelType>('openai');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive or stream updates content
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    const question = inputValue.trim();
    if (!question || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    const assistantId = `${Date.now()}-assistant`;

    setMessages((prev) => [
      ...prev,
      userMessage,
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      },
    ]);
    setInputValue('');
    setIsLoading(true);

    try {
      await streamChatMessage(
        question,
        selectedModel,
        (delta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + delta } : m
            )
          );
        },
        ({ sources, model_used }) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    sources,
                    modelUsed: model_used,
                    isStreaming: false,
                  }
                : m
            )
          );
        }
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to get response';
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: m.content ? `${m.content}\n\nError: ${msg}` : `Error: ${msg}`,
                isError: true,
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <main className="flex flex-col h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 px-6 py-3 shadow-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2.5 rounded-xl shadow-lg">
              <GraduationCap className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Get My University
              </h1>
              <p className="text-xs text-gray-500 font-medium">NEET UG 2026 Assistant</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-full">
              <Shield className="w-4 h-4 text-green-600" />
              <span className="text-xs font-medium text-green-700">Official NTA Source</span>
            </div>
            
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span className="hidden sm:inline">New Chat</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-auto">
        {messages.length === 0 ? (
          // Welcome Screen
          <div className="min-h-full flex flex-col items-center justify-center text-center px-4 py-8">
            {/* Hero Section */}
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-indigo-400 rounded-full blur-2xl opacity-20 animate-pulse" />
              <div className="relative bg-gradient-to-br from-blue-600 to-indigo-600 p-5 rounded-2xl shadow-xl">
                <Sparkles className="w-14 h-14 text-white" />
              </div>
            </div>
            
            <h2 className="text-3xl md:text-4xl font-bold text-gray-800 mb-3">
              NEET UG 2026 <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">AI Assistant</span>
            </h2>
            <p className="text-gray-600 max-w-xl mb-4 text-lg">
              Your trusted companion for all NEET UG 2026 queries. Get instant, accurate answers from the official NTA Information Bulletin.
            </p>
            
            {/* Trust Badges */}
            <div className="flex flex-wrap justify-center gap-3 mb-8">
              <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full">
                <BookOpen className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-700">Official NTA Document</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full">
                <Shield className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-700">100% Authentic Info</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-purple-50 border border-purple-200 rounded-full">
                <Sparkles className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-medium text-purple-700">AI Powered</span>
              </div>
            </div>

            {/* Quick Questions */}
            <div className="w-full max-w-3xl">
              <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
                Popular Questions
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <QuickQuestion
                  icon={<HelpCircle className="w-5 h-5" />}
                  onClick={() => setInputValue('What is the eligibility criteria for NEET UG 2026?')}
                >
                  What is the eligibility criteria for NEET UG 2026?
                </QuickQuestion>
                <QuickQuestion
                  icon={<Calendar className="w-5 h-5" />}
                  onClick={() => setInputValue('What are the important dates for NEET UG 2026?')}
                >
                  What are the important dates for NEET UG 2026?
                </QuickQuestion>
                <QuickQuestion
                  icon={<FileCheck className="w-5 h-5" />}
                  onClick={() => setInputValue('What documents are required for NEET registration?')}
                >
                  What documents are required for registration?
                </QuickQuestion>
                <QuickQuestion
                  icon={<BookOpen className="w-5 h-5" />}
                  onClick={() => setInputValue('What is the pattern for the test?')}
                >
                  What is the pattern for the test?
                </QuickQuestion>
              </div>
            </div>

            {/* Disclaimer */}
            <p className="text-xs text-gray-400 mt-8 max-w-lg">
              Answers are generated from the official NEET UG 2026 Information Bulletin by NTA. 
              Always verify critical information from the official NTA website.
            </p>
          </div>
        ) : (
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            messagesEndRef={messagesEndRef}
          />
        )}
      </div>

      {/* Input Area */}
      <div className="bg-white/80 backdrop-blur-md border-t border-blue-100 px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask any question about NEET UG 2026..."
                className="w-full px-5 py-4 border border-gray-200 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-sm bg-white text-gray-800 placeholder:text-gray-400"
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <LoadingSpinner />
                  <span className="hidden sm:inline">Searching...</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span className="hidden sm:inline">Ask</span>
                </>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            Powered by <span className="font-semibold text-blue-600">Get My University</span> • Press Enter to send
          </p>
        </div>
      </div>
    </main>
  );
}

function QuickQuestion({
  children,
  onClick,
  icon,
}: {
  children: React.ReactNode;
  onClick: () => void;
  icon: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl text-left hover:border-blue-300 hover:bg-blue-50 hover:shadow-md transition-all"
    >
      <div className="p-2 bg-blue-100 rounded-lg text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
        {icon}
      </div>
      <span className="text-gray-700 text-sm font-medium group-hover:text-blue-700">{children}</span>
    </button>
  );
}

function LoadingSpinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
