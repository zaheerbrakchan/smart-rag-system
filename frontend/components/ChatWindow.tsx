'use client';

import { RefObject } from 'react';
import MessageBubble from './MessageBubble';
import { Message } from '@/types';
import { GraduationCap, Search } from 'lucide-react';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  messagesEndRef: RefObject<HTMLDivElement>;
}

export default function ChatWindow({
  messages,
  isLoading,
  messagesEndRef,
}: ChatWindowProps) {
  return (
    <div className="h-full overflow-y-auto px-4 py-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex items-start gap-4 message-enter">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center flex-shrink-0 shadow-md">
              <GraduationCap className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-semibold text-blue-600 mb-1.5">NEET Assistant</p>
              <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm p-4 shadow-md">
                <TypingIndicator />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <Search className="w-4 h-4 text-blue-500 animate-pulse" />
      <span className="text-sm text-gray-500">Searching NEET UG 2026 Bulletin...</span>
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
      </div>
      <span className="text-sm text-gray-500 ml-2">Thinking...</span>
    </div>
  );
}
