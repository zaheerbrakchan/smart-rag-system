'use client';

import { RefObject, useEffect, useMemo, useState } from 'react';
import MessageBubble from './MessageBubble';
import { Message } from '@/types';
import { GraduationCap, Search } from 'lucide-react';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  messagesEndRef: RefObject<HTMLDivElement>;
  onSuggestedReply: (reply: string) => void;
  onCutoffProfileSubmit: (payload: { state: string; category: string; subCategory?: string }) => void;
  language: 'en' | 'hi' | 'mr';
  referencesEnabledGlobal?: boolean;
}

export default function ChatWindow({
  messages,
  isLoading,
  messagesEndRef,
  onSuggestedReply,
  onCutoffProfileSubmit,
  language,
  referencesEnabledGlobal = true,
}: ChatWindowProps) {
  const assistantLabel = language === 'hi' ? 'NEET सहायक' : language === 'mr' ? 'NEET सहाय्यक' : 'NEET Assistant';
  const loadingPhases = useMemo(() => {
    if (language === 'hi') {
      return [
        'आपके प्रश्न का विश्लेषण कर रहा है...',
        'आधिकारिक दस्तावेज़ों और नॉलेज बेस में विवरण खोज रहा है...',
        'आपके लिए सबसे सटीक उत्तर तैयार कर रहा है...',
      ];
    }
    if (language === 'mr') {
      return [
        'तुमच्या प्रश्नाचे विश्लेषण करत आहे...',
        'अधिकृत कागदपत्रे आणि नॉलेज बेसमध्ये तपशील शोधत आहे...',
        'तुमच्यासाठी सर्वात अचूक उत्तर तयार करत आहे...',
      ];
    }
    return [
      'Analyzing your question for accurate guidance...',
      'Searching official documents and our knowledge base for the right details...',
      'Preparing the most accurate answer for you...',
    ];
  }, [language]);
  // Check if the last assistant message has content (means it's streaming)
  const lastMessage = messages[messages.length - 1];
  const isStreaming = lastMessage?.role === 'assistant' && lastMessage?.content?.length > 0;
  
  // Only show loading indicator if loading AND not already streaming content
  const showLoadingIndicator = isLoading && !isStreaming;

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden px-2 md:px-4 py-4 md:py-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onSuggestedReply={onSuggestedReply}
            onCutoffProfileSubmit={onCutoffProfileSubmit}
            language={language}
            referencesEnabledGlobal={referencesEnabledGlobal}
          />
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
                <TypingIndicator loadingPhases={loadingPhases} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

function TypingIndicator({ loadingPhases }: { loadingPhases: string[] }) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const activeLabel = loadingPhases[Math.min(phaseIndex, loadingPhases.length - 1)] || 'Loading...';

  useEffect(() => {
    setPhaseIndex(0);
    const firstTimer = window.setTimeout(() => setPhaseIndex(1), 2000);
    const secondTimer = window.setTimeout(() => setPhaseIndex(2), 5000);
    return () => {
      window.clearTimeout(firstTimer);
      window.clearTimeout(secondTimer);
    };
  }, [loadingPhases]);

  return (
    <div className="flex items-center gap-3">
      <Search className="w-4 h-4 text-blue-500 dark:text-blue-400 animate-pulse" />
      <span className="text-sm text-gray-500 dark:text-gray-400">{activeLabel}</span>
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-blue-400 rounded-full typing-dot" />
      </div>
    </div>
  );
}
