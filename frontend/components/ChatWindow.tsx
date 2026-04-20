'use client';

import { RefObject } from 'react';
import MessageBubble from './MessageBubble';
import { Message } from '@/types';
import { GraduationCap, Search } from 'lucide-react';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  messagesEndRef: RefObject<HTMLDivElement>;
  onSuggestedReply: (reply: string) => void;
  language: 'en' | 'hi' | 'mr';
}

export default function ChatWindow({
  messages,
  isLoading,
  messagesEndRef,
  onSuggestedReply,
  language,
}: ChatWindowProps) {
  const assistantLabel = language === 'hi' ? 'NEET सहायक' : language === 'mr' ? 'NEET सहाय्यक' : 'NEET Assistant';
  const searchingLabel =
    language === 'hi'
      ? 'NEET UG 2026 बुलेटिन खोज रहा है...'
      : language === 'mr'
      ? 'NEET UG 2026 बुलेटिन शोधत आहे...'
      : 'Searching NEET UG 2026 Bulletin...';
  const thinkingLabel = language === 'hi' ? 'सोच रहा है...' : language === 'mr' ? 'विचार करत आहे...' : 'Thinking...';
  // Check if the last assistant message has content (means it's streaming)
  const lastMessage = messages[messages.length - 1];
  const isStreaming = lastMessage?.role === 'assistant' && lastMessage?.content?.length > 0;
  
  // Only show loading indicator if loading AND not already streaming content
  const showLoadingIndicator = isLoading && !isStreaming;

  return (
    <div className="h-full overflow-y-auto px-4 py-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} onSuggestedReply={onSuggestedReply} language={language} />
        ))}

        {/* Loading indicator - only show when waiting for first response */}
        {showLoadingIndicator && (
          <div className="flex items-start gap-4 message-enter">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center flex-shrink-0 shadow-md">
              <GraduationCap className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1.5">{assistantLabel}</p>
              <div className="bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700 rounded-2xl rounded-tl-sm p-4 shadow-md dark:shadow-lg dark:shadow-black/10">
                <TypingIndicator searchingLabel={searchingLabel} thinkingLabel={thinkingLabel} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

function TypingIndicator({ searchingLabel, thinkingLabel }: { searchingLabel: string; thinkingLabel: string }) {
  return (
    <div className="flex items-center gap-3">
      <Search className="w-4 h-4 text-blue-500 dark:text-blue-400 animate-pulse" />
      <span className="text-sm text-gray-500 dark:text-gray-400">{searchingLabel}</span>
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
      </div>
      <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">{thinkingLabel}</span>
    </div>
  );
}
