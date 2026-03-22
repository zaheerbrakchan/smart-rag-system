'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message, Source } from '@/types';
import { User, GraduationCap, ChevronDown, ChevronUp, FileText, AlertCircle, CheckCircle2, BookOpen } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-4 message-enter ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${
          isUser
            ? 'bg-gradient-to-br from-gray-700 to-gray-800'
            : message.isError
            ? 'bg-gradient-to-br from-red-500 to-red-600'
            : 'bg-gradient-to-br from-blue-600 to-indigo-600'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : message.isError ? (
          <AlertCircle className="w-5 h-5 text-white" />
        ) : (
          <GraduationCap className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        {/* Role label */}
        <p className={`text-xs font-semibold mb-1.5 ${isUser ? 'text-gray-500' : 'text-blue-600'}`}>
          {isUser ? 'You' : 'NEET Assistant'}
        </p>
        
        <div
          className={`inline-block rounded-2xl p-4 shadow-sm ${
            isUser
              ? 'bg-gradient-to-br from-gray-700 to-gray-800 text-white rounded-tr-sm'
              : message.isError
              ? 'bg-red-50 border border-red-200 text-red-800 rounded-tl-sm'
              : 'bg-white border border-gray-100 text-gray-800 rounded-tl-sm shadow-md'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : message.isStreaming && !message.content.trim() ? (
            <div className="flex items-center gap-3 py-1">
              <span className="text-sm text-gray-500">Thinking…</span>
              <span className="inline-flex gap-1">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" />
              </span>
            </div>
          ) : message.isStreaming ? (
            <p className="whitespace-pre-wrap text-gray-800 leading-relaxed">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none prose-headings:text-gray-800 prose-p:text-gray-700 prose-li:text-gray-700 prose-strong:text-gray-800">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Source indicator for assistant */}
        {!isUser && !message.isError && !message.isStreaming && (
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-green-50 border border-green-200 rounded-full">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
              <span className="text-xs font-medium text-green-700">Verified from Official Document</span>
            </div>
          </div>
        )}

        {/* Sources section */}
        {!isUser && !message.isStreaming && message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors font-medium"
            >
              <BookOpen className="w-4 h-4" />
              <span>View {message.sources.length} Reference(s) from NEET Bulletin</span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showSources && (
              <div className="mt-3 space-y-2">
                {message.sources.map((source, index) => (
                  <SourceCard key={index} source={source} index={index} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Timestamp */}
        <p className={`text-xs text-gray-400 mt-2 ${isUser ? 'text-right' : ''}`}>
          {formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="p-1.5 bg-blue-100 rounded-lg">
          <FileText className="w-4 h-4 text-blue-600" />
        </div>
        <span className="text-sm font-semibold text-gray-800">
          {source.file_name}
        </span>
        {source.page && (
          <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
            Page {source.page}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-600 leading-relaxed line-clamp-3">{source.text_snippet}</p>
    </div>
  );
}

function formatTime(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}
