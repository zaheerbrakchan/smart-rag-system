'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, Plus, Trash2, Edit2, Check, X, 
  ChevronLeft, ChevronRight, Clock, MoreHorizontal 
} from 'lucide-react';
import { 
  getConversations, 
  deleteConversation, 
  updateConversation,
  ConversationSummary 
} from '@/services/api';

interface ChatSidebarProps {
  token: string;
  currentConversationId: number | null;
  onSelectConversation: (id: number) => void;
  onNewChat: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  language: 'en' | 'hi' | 'mr';
}

export default function ChatSidebar({
  token,
  currentConversationId,
  onSelectConversation,
  onNewChat,
  isCollapsed,
  onToggleCollapse,
  language,
}: ChatSidebarProps) {
  const PAGE_SIZE = 20;
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [totalConversations, setTotalConversations] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [menuOpenId, setMenuOpenId] = useState<number | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const listContainerRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpenId(null);
      }
    };
    
    if (menuOpenId !== null) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [menuOpenId]);

  // Fetch conversations on mount and when token changes
  useEffect(() => {
    if (token) {
      fetchConversations(1, false);
    }
  }, [token]);

  const fetchConversations = async (page: number, append: boolean) => {
    try {
      if (append) {
        setIsLoadingMore(true);
      } else {
        setIsLoading(true);
      }
      const response = await getConversations(token, page, PAGE_SIZE);
      setTotalConversations(response.total || 0);
      setCurrentPage(response.page || page);
      setHasMore((response.page || page) * (response.page_size || PAGE_SIZE) < (response.total || 0));
      setConversations((prev) => {
        if (!append) return response.conversations;
        const seen = new Set(prev.map((c) => c.id));
        const merged = [...prev];
        for (const conv of response.conversations) {
          if (!seen.has(conv.id)) merged.push(conv);
        }
        return merged;
      });
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
    } finally {
      if (append) {
        setIsLoadingMore(false);
      } else {
        setIsLoading(false);
      }
    }
  };

  const handleLoadMore = async () => {
    if (isLoadingMore || isLoading || !hasMore) return;
    await fetchConversations(currentPage + 1, true);
  };

  const handleListScroll = () => {
    const node = listContainerRef.current;
    if (!node || isLoadingMore || isLoading || !hasMore) return;
    const nearBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 120;
    if (nearBottom) {
      void handleLoadMore();
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(id);
    setMenuOpenId(null);
  };

  const confirmDelete = async () => {
    if (!deleteConfirmId) return;
    
    try {
      await deleteConversation(token, deleteConfirmId);
      setConversations(prev => prev.filter(c => c.id !== deleteConfirmId));
      if (currentConversationId === deleteConfirmId) {
        onNewChat();
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
    setDeleteConfirmId(null);
  };

  const cancelDelete = () => {
    setDeleteConfirmId(null);
  };

  const handleStartEdit = (conv: ConversationSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title || `Chat ${conv.id}`);
    setMenuOpenId(null);
  };

  const handleSaveEdit = async (id: number) => {
    try {
      await updateConversation(token, id, editTitle);
      setConversations(prev => 
        prev.map(c => c.id === id ? { ...c, title: editTitle } : c)
      );
    } catch (error) {
      console.error('Failed to update conversation:', error);
    }
    setEditingId(null);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  const i18n = {
    expandSidebar: language === 'hi' ? 'साइडबार खोलें' : language === 'mr' ? 'साइडबार उघडा' : 'Expand sidebar',
    collapseSidebar: language === 'hi' ? 'साइडबार बंद करें' : language === 'mr' ? 'साइडबार बंद करा' : 'Collapse sidebar',
    newChat: language === 'hi' ? 'नई चैट' : language === 'mr' ? 'नवीन चॅट' : 'New Chat',
    noConversations: language === 'hi' ? 'अभी कोई बातचीत नहीं' : language === 'mr' ? 'अजून संभाषणे नाहीत' : 'No conversations yet',
    clickNewChat:
      language === 'hi'
        ? '"नई चैट" पर क्लिक करके शुरू करें'
        : language === 'mr'
        ? '"नवीन चॅट" क्लिक करून सुरू करा'
        : 'Click "New Chat" to start',
    rename: language === 'hi' ? 'नाम बदलें' : language === 'mr' ? 'नाव बदला' : 'Rename',
    delete: language === 'hi' ? 'हटाएं' : language === 'mr' ? 'हटवा' : 'Delete',
    chats: language === 'hi' ? 'चैट' : language === 'mr' ? 'चॅट' : 'chat',
    chatsPlural: language === 'hi' ? 'चैट' : language === 'mr' ? 'चॅट्स' : 'chats',
    msgs: language === 'hi' ? 'संदेश' : language === 'mr' ? 'संदेश' : 'msgs',
    today: language === 'hi' ? 'आज' : language === 'mr' ? 'आज' : 'Today',
    yesterday: language === 'hi' ? 'कल' : language === 'mr' ? 'काल' : 'Yesterday',
    daysAgo:
      language === 'hi'
        ? (d: number) => `${d} दिन पहले`
        : language === 'mr'
        ? (d: number) => `${d} दिवसांपूर्वी`
        : (d: number) => `${d} days ago`,
    deleteConversation: language === 'hi' ? 'बातचीत हटाएं' : language === 'mr' ? 'संभाषण हटवा' : 'Delete Conversation',
    cannotUndo:
      language === 'hi'
        ? 'यह क्रिया वापस नहीं की जा सकती'
        : language === 'mr'
        ? 'ही क्रिया परत आणता येणार नाही'
        : 'This action cannot be undone',
    deleteConfirmBody:
      language === 'hi'
        ? 'क्या आप वाकई यह बातचीत हटाना चाहते हैं? सभी संदेश स्थायी रूप से हट जाएंगे।'
        : language === 'mr'
        ? 'हे संभाषण हटवायचे याची खात्री आहे का? सर्व संदेश कायमचे हटवले जातील.'
        : 'Are you sure you want to delete this conversation? All messages will be permanently removed.',
    cancel: language === 'hi' ? 'रद्द करें' : language === 'mr' ? 'रद्द करा' : 'Cancel',
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return i18n.today;
    if (diffDays === 1) return i18n.yesterday;
    if (diffDays < 7) return i18n.daysAgo(diffDays);
    return date.toLocaleDateString(language === 'hi' ? 'hi-IN' : language === 'mr' ? 'mr-IN' : 'en-US');
  };

  const getTitle = (conv: ConversationSummary) => {
    if (conv.title) return conv.title;
    const chatLabel = language === 'hi' ? 'चैट' : language === 'mr' ? 'चॅट' : 'Chat';
    return `${chatLabel} ${conv.id}`;
  };

  // Collapsed state - just show toggle button
  if (isCollapsed) {
    return (
      <div className="hidden md:flex w-14 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-700/50 flex-col items-center py-4 gap-3">
        <button
          onClick={onToggleCollapse}
          className="p-2.5 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-all"
          title={i18n.expandSidebar}
        >
          <ChevronRight size={20} />
        </button>
        <button
          onClick={onNewChat}
          className="p-2.5 bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-xl shadow-lg hover:shadow-blue-500/25 transition-all hover:scale-105"
          title={i18n.newChat}
        >
          <Plus size={20} />
        </button>
      </div>
    );
  }

  return (
    <>
      <div className="md:hidden fixed inset-0 bg-black/50 z-30" onClick={onToggleCollapse} />
      <div className="fixed inset-y-0 left-0 z-40 w-[86vw] max-w-72 md:static md:w-72 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-700/50 flex flex-col h-full shadow-2xl md:shadow-none">
      {/* Header */}
      <div className="p-3 md:p-4 border-b border-slate-200 dark:border-slate-700/50">
        <div className="flex items-center gap-2">
          <button
            onClick={onNewChat}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl transition-all font-medium text-sm shadow-lg hover:shadow-blue-500/25"
          >
            <Plus size={18} />
            {i18n.newChat}
          </button>
          <button
            onClick={onToggleCollapse}
            className="p-3 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors"
            title={i18n.collapseSidebar}
          >
            <ChevronLeft size={18} />
          </button>
        </div>
      </div>

      {/* Conversation List */}
      <div
        ref={listContainerRef}
        onScroll={handleListScroll}
        className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent"
      >
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-12 px-6">
            <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="w-8 h-8 text-slate-500 dark:text-slate-500" />
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">{i18n.noConversations}</p>
            <p className="text-slate-500 dark:text-slate-600 text-xs mt-1">{i18n.clickNewChat}</p>
          </div>
        ) : (
          <div className="py-2 space-y-1">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group relative mx-2 rounded-lg transition-all ${
                  currentConversationId === conv.id
                    ? 'bg-gradient-to-r from-blue-600/20 to-indigo-600/20 border border-blue-500/30'
                    : 'hover:bg-slate-100 dark:hover:bg-slate-800/80'
                }`}
              >
                {editingId === conv.id ? (
                  <div className="flex items-center gap-2 p-2" onClick={e => e.stopPropagation()}>
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      className="flex-1 bg-white dark:bg-slate-700 text-slate-900 dark:text-white text-sm px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit(conv.id);
                        if (e.key === 'Escape') handleCancelEdit();
                      }}
                    />
                    <button
                      onClick={() => handleSaveEdit(conv.id)}
                      className="p-2 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
                    >
                      <Check size={14} />
                    </button>
                    <button
                      onClick={handleCancelEdit}
                      className="p-2 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <div 
                    className="flex items-center gap-3 p-3 cursor-pointer"
                    onClick={() => onSelectConversation(conv.id)}
                  >
                    <div className={`p-2 rounded-lg ${
                      currentConversationId === conv.id 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 group-hover:bg-slate-200 dark:group-hover:bg-slate-600 group-hover:text-slate-700 dark:group-hover:text-slate-300'
                    }`}>
                      <MessageSquare size={16} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium truncate ${
                        currentConversationId === conv.id ? 'text-white' : 'text-slate-700 dark:text-slate-200'
                      }`}>
                        {getTitle(conv)}
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-500 flex items-center gap-1.5 mt-0.5">
                        <Clock size={10} />
                        {formatDate(conv.updated_at)}
                        <span className="text-slate-400 dark:text-slate-600">•</span>
                        {conv.message_count} {i18n.msgs}
                      </p>
                    </div>
                    
                    {/* Action menu button - always visible on hover or when menu is open */}
                    <div 
                      ref={menuOpenId === conv.id ? menuRef : null}
                      className={`relative ${menuOpenId === conv.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'} transition-opacity`}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setMenuOpenId(menuOpenId === conv.id ? null : conv.id);
                        }}
                        className={`p-1.5 rounded-lg transition-colors ${
                          menuOpenId === conv.id 
                            ? 'bg-slate-200 dark:bg-slate-600 text-slate-900 dark:text-white' 
                            : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-600'
                        }`}
                      >
                        <MoreHorizontal size={16} />
                      </button>
                      
                      {/* Dropdown Menu */}
                      {menuOpenId === conv.id && (
                        <div className="absolute right-0 top-full mt-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-xl shadow-xl py-1.5 z-50 min-w-[140px] overflow-hidden">
                          <button
                            onClick={(e) => handleStartEdit(conv, e)}
                            className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-slate-900 dark:hover:text-white transition-colors"
                          >
                            <Edit2 size={14} className="text-blue-400" />
                            {i18n.rename}
                          </button>
                          <div className="h-px bg-slate-200 dark:bg-slate-700 mx-2 my-1" />
                          <button
                            onClick={(e) => handleDelete(conv.id, e)}
                            className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-red-400 hover:bg-red-900/30 hover:text-red-300 transition-colors"
                          >
                            <Trash2 size={14} />
                            {i18n.delete}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            {hasMore && (
              <div className="px-3 pt-2 pb-3">
                <button
                  onClick={() => void handleLoadMore()}
                  disabled={isLoadingMore}
                  className="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800/70 text-slate-700 dark:text-slate-200 text-xs py-2 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-60"
                >
                  {isLoadingMore ? 'Loading more...' : 'Load more'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-700/50">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-500">
          <span>
            {conversations.length}
            {totalConversations > conversations.length ? `/${totalConversations}` : ''}{' '}
            {totalConversations !== 1 ? i18n.chatsPlural : i18n.chats}
          </span>
          <span className="text-slate-500 dark:text-slate-600">NEET UG 2026</span>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={cancelDelete}
          />
          {/* Modal */}
          <div className="relative bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-2xl shadow-2xl p-6 max-w-sm mx-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 bg-red-500/20 rounded-xl">
                <Trash2 className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">{i18n.deleteConversation}</h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">{i18n.cannotUndo}</p>
              </div>
            </div>
            <p className="text-slate-700 dark:text-slate-300 text-sm mb-6">
              {i18n.deleteConfirmBody}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={cancelDelete}
                className="px-4 py-2.5 text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded-xl transition-colors"
              >
                {i18n.cancel}
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2.5 text-sm font-medium text-white bg-red-600 hover:bg-red-500 rounded-xl transition-colors shadow-lg shadow-red-500/25"
              >
                {i18n.delete}
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </>
  );
}
