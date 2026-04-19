'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import ChatWindow from '@/components/ChatWindow';
import ChatSidebar from '@/components/ChatSidebar';
import ThemeToggle from '@/components/ThemeToggle';
import { Message, ModelType } from '@/types';
import { streamChatMessage, UserPreferences, getConversation } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { 
  GraduationCap, Send, Sparkles, BookOpen, Calendar, FileCheck, 
  HelpCircle, Shield, RotateCcw, Settings, LogIn, UserPlus, 
  User, LogOut, ChevronDown, Menu 
} from 'lucide-react';

type GuidedIntent =
  | 'neet_exam_guidance'
  | 'counselling_process'
  | 'college_shortlist'
  | 'college_fee_structure';

const STARTER_INTENT_MAP: Record<string, GuidedIntent> = {
  'NEET exam guidance': 'neet_exam_guidance',
  'Counselling process': 'counselling_process',
  'College shortlist': 'college_shortlist',
  'College fee structure': 'college_fee_structure',
};

const GUIDED_PROMPTS: Record<GuidedIntent, { message: string }> = {
  neet_exam_guidance: {
    message:
      'Great choice. What would you like to know in NEET exam guidance?',
  },
  counselling_process: {
    message:
      'Sure — please type the state/UT counselling details you want to know.',
  },
  college_shortlist: {
    message:
      'To shortlist accurately, please share your NEET rank (or expected rank), category, and preferred state.',
  },
  college_fee_structure: {
    message:
      'Sure — tell me which state or college fee structure you want, and if possible mention college type.',
  },
};

function buildGuidedQuestion(intent: GuidedIntent, userReply: string): string {
  const detail = userReply.trim();
  switch (intent) {
    case 'neet_exam_guidance':
      return `For NEET UG 2026, explain ${detail} in a clear, student-friendly way.`;
    case 'counselling_process':
      return `Explain NEET UG counselling process for ${detail}, including registration steps, key dates, required documents, and round-wise flow.`;
    case 'college_shortlist':
      return `Help me with college shortlisting based on this profile: ${detail}. Ask one follow-up only if essential details are missing.`;
    case 'college_fee_structure':
      return `Show college fee structure details for this preference: ${detail}. Include what is available in official counselling documents.`;
    default:
      return detail;
  }
}

export default function Home() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: authLoading, logout, token } = useAuth();
  const { theme } = useTheme();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedModel, setSelectedModel] = useState<ModelType>('openai');
  const [isLoading, setIsLoading] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [pendingClarification, setPendingClarification] = useState<{question: string, messageId: string} | null>(null);
  const [guidedIntent, setGuidedIntent] = useState<GuidedIntent | null>(null);
  const [allowStarterReplies, setAllowStarterReplies] = useState(true);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarKey, setSidebarKey] = useState(0); // To refresh sidebar
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.replace('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (quickReply?: string) => {
    const trimmed = (quickReply ?? inputValue).trim();
    const pending = pendingClarification;

    if (isLoading) return;

    if (!pending && quickReply && STARTER_INTENT_MAP[quickReply]) {
      const intent = STARTER_INTENT_MAP[quickReply];
      const guide = GUIDED_PROMPTS[intent];
      const turnId = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      const userMessage: Message = {
        id: `user-${turnId}`,
        role: 'user',
        content: quickReply,
        timestamp: new Date(),
      };
      const assistantMessage: Message = {
        id: `assistant-${turnId}`,
        role: 'assistant',
        content: guide.message,
        timestamp: new Date(),
      };
      setGuidedIntent(intent);
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setInputValue('');
      return;
    }

    let question: string;
    let clarifiedScope: string | undefined;
    let assistantMessageId: string;

    if (pending) {
      if (!trimmed) return;
      question = pending.question;
      clarifiedScope = trimmed;
      assistantMessageId = pending.messageId;

      const userMessage: Message = {
        id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setInputValue('');
      setPendingClarification(null);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: '', needsClarification: false, clarificationOptions: undefined, suggestedReplies: undefined }
            : msg
        )
      );
    } else {
      if (!trimmed) return;
      question = guidedIntent ? buildGuidedQuestion(guidedIntent, trimmed) : trimmed;
      if (guidedIntent) {
        setGuidedIntent(null);
      }
      clarifiedScope = undefined;
      // Must never collide: (Date.now()+1) then Date.now() can equal the assistant id if the clock ticks 1ms.
      const turnId = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      const userMessage: Message = {
        id: `user-${turnId}`,
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      };
      assistantMessageId = `assistant-${turnId}`;
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        sources: [],
      };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setInputValue('');
    }

    setIsLoading(true);

    try {
      // Build user preferences for smart routing
      const userPreferences: UserPreferences | undefined = user?.preferences ? {
        preferred_state: user.preferences.preferred_state,
        category: user.preferences.category,
      } : undefined;
      
      // Stream response from API
      await streamChatMessage(
        question,
        selectedModel,
        // onToken - append each token
        (token) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, content: msg.content + token }
                : msg
            )
          );
        },
        // onSources - set sources
        (sources) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, sources }
                : msg
            )
          );
        },
        // onDone - update model and conversation ID
        (filters, newConversationId) => {
          if (newConversationId && newConversationId !== conversationId) {
            setConversationId(newConversationId);
          }
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, modelUsed: selectedModel }
                : msg
            )
          );
        },
        // onError
        (error) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, content: `Error: ${error}`, isError: true }
                : msg
            )
          );
        },
        userPreferences,
        clarifiedScope,
        // onClarificationNeeded
        (_options, message) => {
          setPendingClarification({ question, messageId: assistantMessageId });
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    content: message,
                    needsClarification: true,
                    clarificationOptions: _options || [],
                    suggestedReplies: _options || [],
                    originalQuestion: question,
                  }
                : msg
            )
          );
        },
        // onSuggestedReplies
        (replies) => {
          const normalizedReplies = (replies || [])
            .map((reply) => String(reply || '').trim())
            .filter((reply) => reply.length > 0)
            .slice(0, 6);
          if (normalizedReplies.length === 0) {
            return;
          }

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, suggestedReplies: normalizedReplies }
                : msg
            )
          );
          if (normalizedReplies.some((reply) => Boolean(STARTER_INTENT_MAP[reply]))) {
            setAllowStarterReplies(false);
          }
        },
        // conversationId and userId
        conversationId || undefined,
        user?.id,
        // onTitle - refresh sidebar when title is generated
        (title, convId) => {
          if (convId) {
            setSidebarKey(prev => prev + 1);
          }
        }
      );
    } catch (error) {
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === assistantMessageId 
            ? { ...msg, content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`, isError: true }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedReply = (reply: string) => {
    if (isLoading) return;
    void handleSendMessage(reply);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Load a conversation from sidebar
  const handleSelectConversation = async (id: number) => {
    if (!token || id === conversationId) return;
    
    try {
      setIsLoading(true);
      const conv = await getConversation(token, id);
      
      // Convert API messages to our Message format
      const loadedMessages: Message[] = conv.messages.map((m) => ({
        id: m.id.toString(),
        role: m.role as 'user' | 'assistant',
        content: m.content,
        timestamp: new Date(m.created_at),
        sources: m.sources || undefined,
      }));
      
      setMessages(loadedMessages);
      setConversationId(id);
      setPendingClarification(null);
      setGuidedIntent(null);
      setAllowStarterReplies(loadedMessages.length === 0);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Start new chat
  const handleNewChat = () => {
    setMessages([]);
    setConversationId(null);
    setPendingClarification(null);
    setGuidedIntent(null);
    setAllowStarterReplies(true);
    setInputValue('');
  };

  const clearChat = () => {
    setMessages([]);
    setPendingClarification(null);
    setGuidedIntent(null);
    setAllowStarterReplies(true);
    setConversationId(null);  // Reset conversation for new chat
    setSidebarKey(prev => prev + 1); // Refresh sidebar to show new conversation
  };

  // Show loading only while checking auth status
  if (authLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  // If not authenticated, show nothing (redirect is happening)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Chat History Sidebar */}
      {isAuthenticated && token && (
        <ChatSidebar
          key={sidebarKey}
          token={token}
          currentConversationId={conversationId}
          onSelectConversation={handleSelectConversation}
          onNewChat={handleNewChat}
          isCollapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      )}
      
      <main className="flex flex-col flex-1 bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 overflow-hidden">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-blue-100 dark:border-slate-700 px-6 py-2.5 shadow-sm sticky top-0 z-50 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl shadow-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Med Buddy
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Powered by Get My University</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-full">
              <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
              <span className="text-xs font-medium text-green-700 dark:text-green-400">Official NTA Source</span>
            </div>
            
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {/* Admin Dashboard - Modern Glass Design */}
            {isAuthenticated && (user?.role === 'admin' || user?.role === 'super_admin') && (
              <Link
                href="/admin"
                className="group flex items-center gap-2 px-3 py-2 bg-white/80 dark:bg-slate-700/80 backdrop-blur-xl border border-gray-200/60 dark:border-slate-600 rounded-xl shadow-sm hover:shadow-md hover:border-indigo-300 dark:hover:border-indigo-500 hover:bg-gradient-to-r hover:from-indigo-50 hover:to-purple-50 dark:hover:from-indigo-900/30 dark:hover:to-purple-900/30 transition-all duration-300"
              >
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg blur-sm opacity-0 group-hover:opacity-60 transition-opacity duration-300" />
                  <div className="relative w-6 h-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center shadow-md shadow-indigo-500/30">
                    <Settings className="w-3.5 h-3.5 text-white group-hover:rotate-180 transition-transform duration-500" />
                  </div>
                </div>
                <div className="hidden sm:block">
                  <p className="text-xs font-semibold text-gray-800 group-hover:text-indigo-700 transition-colors">Dashboard</p>
                  <p className="text-[10px] text-gray-400 group-hover:text-indigo-400 transition-colors -mt-0.5">Admin</p>
                </div>
              </Link>
            )}
            
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span className="hidden sm:inline">New Chat</span>
              </button>
            )}
            
            {/* Auth buttons */}
            {!authLoading && (
              <>
                {isAuthenticated ? (
                  <div className="relative">
                    <button
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                    >
                      <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                        {user?.full_name?.charAt(0).toUpperCase() || 'U'}
                      </div>
                      <span className="hidden sm:inline max-w-24 truncate">{user?.full_name?.split(' ')[0]}</span>
                      <ChevronDown className="w-4 h-4" />
                    </button>
                    
                    {showUserMenu && (
                      <>
                        <div className="fixed inset-0 z-40" onClick={() => setShowUserMenu(false)} />
                        <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-100 dark:border-slate-700 py-2 z-50">
                          <div className="px-4 py-3 border-b border-gray-100 dark:border-slate-700">
                            <p className="text-sm font-medium text-gray-800 dark:text-white">{user?.full_name}</p>
                          </div>
                          <Link
                            href="/profile"
                            className="flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-slate-700"
                            onClick={() => setShowUserMenu(false)}
                          >
                            <User className="w-4 h-4" />
                            My Profile
                          </Link>
                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              logout();
                            }}
                            className="w-full flex items-center gap-3 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
                          >
                            <LogOut className="w-4 h-4" />
                            Sign Out
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Link
                      href="/login"
                      className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                    >
                      <LogIn className="w-4 h-4" />
                      <span className="hidden sm:inline">Sign In</span>
                    </Link>
                    <Link
                      href="/register"
                      className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 rounded-lg transition-colors"
                    >
                      <UserPlus className="w-4 h-4" />
                      <span className="hidden sm:inline">Sign Up</span>
                    </Link>
                  </div>
                )}
              </>
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
            
            <h2 className="text-3xl md:text-4xl font-bold text-gray-800 dark:text-white mb-3">
              NEET UG 2026 <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Med Buddy</span>
            </h2>
            <p className="text-gray-600 dark:text-gray-300 max-w-xl mb-4 text-lg">
              India&apos;s counselling companion for NEET UG aspirants. Get structured, reliable guidance on college shortlist, fee structures, NEET exam process, and counselling roadmap.
            </p>
            
            {/* Trust Badges */}
            <div className="flex flex-wrap justify-center gap-3 mb-8">
              <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-full">
                <BookOpen className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-400">Official NTA Document</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-full">
                <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-700 dark:text-green-400">100% Authentic Info</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-purple-50 dark:bg-purple-900/30 border border-purple-200 dark:border-purple-700 rounded-full">
                <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-medium text-purple-700 dark:text-purple-400">AI Powered</span>
              </div>
            </div>

            {/* Quick Questions */}
            <div className="w-full max-w-3xl">
              <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
                Start With
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <QuickQuestion
                  icon={<Calendar className="w-5 h-5" />}
                  onClick={() => void handleSendMessage('NEET exam guidance')}
                >
                  NEET exam guidance
                </QuickQuestion>
                <QuickQuestion
                  icon={<BookOpen className="w-5 h-5" />}
                  onClick={() => void handleSendMessage('Counselling process')}
                >
                  Counselling process
                </QuickQuestion>
                <QuickQuestion
                  icon={<HelpCircle className="w-5 h-5" />}
                  onClick={() => void handleSendMessage('College shortlist')}
                >
                  College shortlist
                </QuickQuestion>
                <QuickQuestion
                  icon={<FileCheck className="w-5 h-5" />}
                  onClick={() => void handleSendMessage('College fee structure')}
                >
                  College fee structure
                </QuickQuestion>
              </div>
            </div>

            {/* Note / disclaimer (empty-state footer) */}
            <p className="text-xs italic text-gray-400 dark:text-gray-500 mt-8 max-w-lg leading-relaxed">
              <span className="font-medium not-italic text-gray-500 dark:text-gray-400">Note: </span>
              Med Buddy is powered by Get My University. Guidance is based on available counselling documents and official sources.
              Always verify final admission decisions with MCC/state counselling authorities and college websites.
            </p>
          </div>
        ) : (
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            messagesEndRef={messagesEndRef}
            onSuggestedReply={handleSuggestedReply}
          />
        )}
      </div>

      {/* Input Area */}
      <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-t border-blue-100 dark:border-slate-700 px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  pendingClarification
                    ? 'Reply in your own words — e.g. All India / MCC, or name a state…'
                    : 'Ask any question about NEET UG 2026...'
                }
                className="w-full px-5 py-4 border border-gray-200 dark:border-slate-600 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-sm bg-white dark:bg-slate-700 text-gray-800 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={() => handleSendMessage()}
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
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">
            Powered by <span className="font-semibold text-blue-600 dark:text-blue-400">Get My University</span> • Press Enter to send
          </p>
        </div>
      </div>
    </main>
    </div>
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
      className="group flex items-center gap-3 p-4 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl text-left hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:shadow-md transition-all"
    >
      <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg text-blue-600 dark:text-blue-400 group-hover:bg-blue-600 group-hover:text-white transition-colors">
        {icon}
      </div>
      <span className="text-gray-700 dark:text-gray-300 text-sm font-medium group-hover:text-blue-700 dark:group-hover:text-blue-400">{children}</span>
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
