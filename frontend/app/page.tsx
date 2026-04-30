'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import ChatWindow from '@/components/ChatWindow';
import ChatSidebar from '@/components/ChatSidebar';
import ThemeToggle from '@/components/ThemeToggle';
import { Message, ModelType } from '@/types';
import {
  streamChatMessage,
  UserPreferences,
  getConversation,
  createSupportQuery,
  getMySupportQueries,
  getMySupportNotifications,
  markSupportNotificationRead,
  connectSupportNotificationStream,
  SupportQuery,
  SupportNotification,
} from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { 
  GraduationCap, Send, Sparkles, BookOpen, Calendar, FileCheck, 
  HelpCircle, Shield, Settings, LogIn, UserPlus, 
  User, LogOut, ChevronDown, Menu, MessageCircleQuestion, Bell
} from 'lucide-react';

/** Same base URL pattern as `services/api.ts` — used for direct fetch calls in this page. */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type GuidedIntent =
  | 'neet_exam_guidance'
  | 'counselling_process'
  | 'college_shortlist'
  | 'college_fee_structure';

type LanguageCode = 'en' | 'hi' | 'mr';

const TRANSLATIONS: Record<
  LanguageCode,
  {
    languageLabel: string;
    poweredByHeader: string;
    officialNtaSource: string;
    dashboard: string;
    admin: string;
    newChat: string;
    myProfile: string;
    signOut: string;
    signIn: string;
    signUp: string;
    heroDescription: string;
    officialNtaDocument: string;
    authenticInfo: string;
    aiPowered: string;
    startWith: string;
    noteLabel: string;
    noteBody: string;
    placeholderClarification: string;
    placeholderDefault: string;
    searching: string;
    ask: string;
    footerPowerBy: string;
    footerEnter: string;
    contactSupport: string;
    mySupportUpdates: string;
    supportModalTitle: string;
    supportMessagePlaceholder: string;
    submitSupport: string;
    supportSent: string;
    close: string;
    notificationsTitle: string;
    supportHistoryTitle: string;
    loading: string;
    noSupportQueries: string;
    noNotifications: string;
    unreadLabel: string;
    statusPending: string;
    statusInProgress: string;
    statusAnswered: string;
    statusClosed: string;
    subjectLabel: string;
    messageLabel: string;
    latestReplyLabel: string;
    markRead: string;
    starter: Record<GuidedIntent, string>;
    guidedPrompts: Record<GuidedIntent, string>;
  }
> = {
  en: {
    languageLabel: 'English',
    poweredByHeader: 'Powered by Get My University',
    officialNtaSource: 'Official NTA Source',
    dashboard: 'Dashboard',
    admin: 'Admin',
    newChat: 'New Chat',
    myProfile: 'My Profile',
    signOut: 'Sign Out',
    signIn: 'Sign In',
    signUp: 'Sign Up',
    heroDescription:
      "India's counselling companion for NEET UG aspirants. Get structured, reliable guidance on college shortlist, fee structures, NEET exam process, and counselling roadmap.",
    officialNtaDocument: 'Official NTA Document',
    authenticInfo: '100% Authentic Info',
    aiPowered: 'AI Powered',
    startWith: 'Start With',
    noteLabel: 'Note: ',
    noteBody:
      'Med Buddy is powered by Get My University. Guidance is based on available counselling documents and official sources. Always verify final admission decisions with MCC/state counselling authorities and college websites.',
    placeholderClarification: 'Reply in your own words — e.g. All India / MCC, or name a state…',
    placeholderDefault: 'Ask any question about NEET UG 2026...',
    searching: 'Searching...',
    ask: 'Ask',
    footerPowerBy: 'Powered by',
    footerEnter: 'Press Enter to send',
    contactSupport: 'Support',
    mySupportUpdates: 'My Queries',
    supportModalTitle: 'Send query to Get My University team',
    supportMessagePlaceholder: 'Write your query/feedback here...',
    submitSupport: 'Send Query',
    supportSent: 'Support query submitted successfully.',
    close: 'Close',
    notificationsTitle: 'Notifications',
    supportHistoryTitle: 'My Submitted Queries',
    loading: 'Loading...',
    noSupportQueries: 'No support queries yet.',
    noNotifications: 'No notifications yet.',
    unreadLabel: 'Unread',
    statusPending: 'Pending',
    statusInProgress: 'In Progress',
    statusAnswered: 'Answered',
    statusClosed: 'Closed',
    subjectLabel: 'Subject',
    messageLabel: 'Message',
    latestReplyLabel: 'Latest reply',
    markRead: 'Mark read',
    starter: {
      neet_exam_guidance: 'NEET Exam Guidance',
      counselling_process: 'Counselling Process',
      college_shortlist: 'College Shortlist',
      college_fee_structure: 'College Fee Structure',
    },
    guidedPrompts: {
      neet_exam_guidance: 'Great choice. What would you like to know in NEET exam guidance?',
      counselling_process: 'Sure — please type the state/UT counselling details you want to know.',
      college_shortlist:
        'Great — I can help with college shortlist. I will quickly collect your needed details and then suggest the best matches. Can you please tell your home state?',
      college_fee_structure:
        'Sure — tell me which state or college fee structure you want, and if possible mention college type.',
    },
  },
  hi: {
    languageLabel: 'हिंदी',
    poweredByHeader: 'Get My University द्वारा संचालित',
    officialNtaSource: 'आधिकारिक NTA स्रोत',
    dashboard: 'डैशबोर्ड',
    admin: 'एडमिन',
    newChat: 'नई चैट',
    myProfile: 'मेरा प्रोफाइल',
    signOut: 'साइन आउट',
    signIn: 'साइन इन',
    signUp: 'साइन अप',
    heroDescription:
      'NEET UG अभ्यर्थियों के लिए काउंसलिंग साथी। कॉलेज शॉर्टलिस्ट, फीस, NEET प्रक्रिया और काउंसलिंग रोडमैप पर भरोसेमंद मार्गदर्शन पाएं।',
    officialNtaDocument: 'आधिकारिक NTA दस्तावेज़',
    authenticInfo: '100% प्रमाणित जानकारी',
    aiPowered: 'AI संचालित',
    startWith: 'शुरुआत करें',
    noteLabel: 'नोट: ',
    noteBody:
      'Med Buddy, Get My University द्वारा संचालित है। मार्गदर्शन उपलब्ध काउंसलिंग दस्तावेज़ों और आधिकारिक स्रोतों पर आधारित है। अंतिम निर्णय से पहले MCC/राज्य काउंसलिंग प्राधिकरण और कॉलेज वेबसाइट पर अवश्य सत्यापित करें।',
    placeholderClarification: 'अपने शब्दों में जवाब दें — जैसे All India / MCC, या किसी राज्य का नाम…',
    placeholderDefault: 'NEET UG 2026 से जुड़ा कोई भी प्रश्न पूछें...',
    searching: 'खोज जारी है...',
    ask: 'पूछें',
    footerPowerBy: 'संचालित द्वारा',
    footerEnter: 'भेजने के लिए Enter दबाएं',
    contactSupport: 'सहायता',
    mySupportUpdates: 'मेरे प्रश्न',
    supportModalTitle: 'Get My University टीम को प्रश्न भेजें',
    supportMessagePlaceholder: 'अपना प्रश्न/फीडबैक यहां लिखें...',
    submitSupport: 'प्रश्न भेजें',
    supportSent: 'सपोर्ट प्रश्न सफलतापूर्वक भेजा गया।',
    close: 'बंद करें',
    notificationsTitle: 'सूचनाएं',
    supportHistoryTitle: 'मेरे भेजे गए प्रश्न',
    loading: 'लोड हो रहा है...',
    noSupportQueries: 'अभी कोई सपोर्ट प्रश्न नहीं है।',
    noNotifications: 'अभी कोई सूचना नहीं है।',
    unreadLabel: 'अपठित',
    statusPending: 'लंबित',
    statusInProgress: 'प्रगति में',
    statusAnswered: 'उत्तर दिया गया',
    statusClosed: 'बंद',
    subjectLabel: 'विषय',
    messageLabel: 'संदेश',
    latestReplyLabel: 'नवीनतम उत्तर',
    markRead: 'पढ़ा हुआ चिन्हित करें',
    starter: {
      neet_exam_guidance: 'NEET परीक्षा मार्गदर्शन',
      counselling_process: 'काउंसलिंग प्रक्रिया',
      college_shortlist: 'कॉलेज शॉर्टलिस्ट',
      college_fee_structure: 'कॉलेज फीस संरचना',
    },
    guidedPrompts: {
      neet_exam_guidance: 'बहुत बढ़िया। NEET परीक्षा मार्गदर्शन में आप क्या जानना चाहते हैं?',
      counselling_process: 'ज़रूर — जिस राज्य/केंद्र शासित प्रदेश की काउंसलिंग जानकारी चाहिए, वह लिखें।',
      college_shortlist:
        'बहुत बढ़िया — मैं कॉलेज शॉर्टलिस्ट में मदद करूंगा। जरूरी जानकारी लेकर सबसे उपयुक्त विकल्प बताऊंगा। कृपया अपना होम स्टेट बताएं।',
      college_fee_structure:
        'ज़रूर — जिस राज्य या कॉलेज की फीस संरचना चाहिए, बताएं; संभव हो तो कॉलेज प्रकार भी लिखें।',
    },
  },
  mr: {
    languageLabel: 'मराठी',
    poweredByHeader: 'Get My University द्वारे समर्थित',
    officialNtaSource: 'अधिकृत NTA स्रोत',
    dashboard: 'डॅशबोर्ड',
    admin: 'ॲडमिन',
    newChat: 'नवीन चॅट',
    myProfile: 'माझे प्रोफाइल',
    signOut: 'साइन आउट',
    signIn: 'साइन इन',
    signUp: 'साइन अप',
    heroDescription:
      'NEET UG विद्यार्थ्यांसाठी समुपदेशन साथी. कॉलेज शॉर्टलिस्ट, फी स्ट्रक्चर, NEET प्रक्रिया आणि समुपदेशन रोडमॅपसाठी विश्वसनीय मार्गदर्शन मिळवा.',
    officialNtaDocument: 'अधिकृत NTA दस्तऐवज',
    authenticInfo: '100% प्रमाणित माहिती',
    aiPowered: 'AI समर्थित',
    startWith: 'यापासून सुरू करा',
    noteLabel: 'टीप: ',
    noteBody:
      'Med Buddy हे Get My University द्वारे समर्थित आहे. मार्गदर्शन उपलब्ध समुपदेशन दस्तऐवज आणि अधिकृत स्रोतांवर आधारित आहे. अंतिम प्रवेश निर्णयापूर्वी MCC/राज्य समुपदेशन प्राधिकरण आणि कॉलेज संकेतस्थळावर पडताळणी करा.',
    placeholderClarification: 'तुमच्या शब्दांत उत्तर द्या — उदा. All India / MCC किंवा राज्याचे नाव…',
    placeholderDefault: 'NEET UG 2026 बद्दल कोणताही प्रश्न विचारा...',
    searching: 'शोध सुरू आहे...',
    ask: 'विचारा',
    footerPowerBy: 'समर्थित',
    footerEnter: 'पाठवण्यासाठी Enter दाबा',
    contactSupport: 'सपोर्ट',
    mySupportUpdates: 'माझे प्रश्न',
    supportModalTitle: 'Get My University टीमला प्रश्न पाठवा',
    supportMessagePlaceholder: 'तुमचा प्रश्न/फीडबॅक येथे लिहा...',
    submitSupport: 'प्रश्न पाठवा',
    supportSent: 'सपोर्ट प्रश्न यशस्वीरित्या पाठवला गेला.',
    close: 'बंद करा',
    notificationsTitle: 'सूचना',
    supportHistoryTitle: 'माझे पाठवलेले प्रश्न',
    loading: 'लोड होत आहे...',
    noSupportQueries: 'अद्याप कोणतेही सपोर्ट प्रश्न नाहीत.',
    noNotifications: 'अद्याप कोणत्याही सूचना नाहीत.',
    unreadLabel: 'न वाचलेले',
    statusPending: 'प्रलंबित',
    statusInProgress: 'प्रगतीत',
    statusAnswered: 'उत्तर दिले',
    statusClosed: 'बंद',
    subjectLabel: 'विषय',
    messageLabel: 'संदेश',
    latestReplyLabel: 'नवीन उत्तर',
    markRead: 'वाचले म्हणून चिन्हांकित करा',
    starter: {
      neet_exam_guidance: 'NEET परीक्षा मार्गदर्शन',
      counselling_process: 'समुपदेशन प्रक्रिया',
      college_shortlist: 'कॉलेज शॉर्टलिस्ट',
      college_fee_structure: 'कॉलेज फी संरचना',
    },
    guidedPrompts: {
      neet_exam_guidance: 'छान निवड. NEET परीक्षा मार्गदर्शनात तुम्हाला काय जाणून घ्यायचे आहे?',
      counselling_process: 'नक्की — ज्या राज्य/केंद्रशासित प्रदेशाची समुपदेशन माहिती हवी आहे ती टाइप करा.',
      college_shortlist:
        'छान — मी कॉलेज शॉर्टलिस्टमध्ये मदत करीन. आवश्यक माहिती घेऊन योग्य पर्याय सुचवेन. कृपया तुमचे होम स्टेट सांगा.',
      college_fee_structure:
        'नक्की — कोणत्या राज्य/कॉलेजची फी संरचना हवी ते सांगा; शक्य असल्यास कॉलेज प्रकारही नमूद करा.',
    },
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageCode>('en');
  const [showSupportModal, setShowSupportModal] = useState(false);
  const [showSupportPanel, setShowSupportPanel] = useState(false);
  const [supportMessage, setSupportMessage] = useState('');
  const [supportLoading, setSupportLoading] = useState(false);
  const [supportSuccess, setSupportSuccess] = useState<string | null>(null);
  const [supportError, setSupportError] = useState<string | null>(null);
  const [mySupportQueries, setMySupportQueries] = useState<SupportQuery[]>([]);
  const [myNotifications, setMyNotifications] = useState<SupportNotification[]>([]);
  const [supportUnreadCount, setSupportUnreadCount] = useState(0);
  const [supportPanelLoading, setSupportPanelLoading] = useState(false);
  const [chatReferencesEnabledGlobal, setChatReferencesEnabledGlobal] = useState(true);
  const [sidebarKey, setSidebarKey] = useState(0); // To refresh sidebar
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const conversationIdRef = useRef<number | null>(null);
  const t = TRANSLATIONS[selectedLanguage];
  const isAdminUser = user?.role === 'admin' || user?.role === 'super_admin';
  const displayedMessages = useMemo(
    () =>
      messages.map((msg) => {
        if (chatReferencesEnabledGlobal || msg.role !== 'assistant') return msg;
        const inferredOrigin =
          msg.sourceOrigin ??
          ((msg.sources || []).some((s: any) => {
            const dtype = String(s?.document_type || '').toLowerCase();
            const name = String(s?.file_name || '').toLowerCase();
            return dtype.includes('web') || name.startsWith('http://') || name.startsWith('https://');
          })
            ? 'web'
            : (msg.sources || []).length > 0
            ? 'kb'
            : undefined);
        return {
          ...msg,
          sourceOrigin: inferredOrigin,
          sources: undefined,
          referencesEnabled: false,
        };
      }),
    [messages, chatReferencesEnabledGlobal]
  );

  const normalizeGuidedText = (text: string): string =>
    text
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, ' ')
      .replace(/\s+/g, ' ');

  const resolveGuidedIntent = (reply: string): GuidedIntent | null => {
    const val = normalizeGuidedText(reply);
    const langs: LanguageCode[] = ['en', 'hi', 'mr'];
    for (const lang of langs) {
      const entries = Object.entries(TRANSLATIONS[lang].starter) as Array<[GuidedIntent, string]>;
      for (const [intent, label] of entries) {
        if (val === normalizeGuidedText(label)) return intent;
      }
    }
    return null;
  };

  const supportStatusLabel = (status: string) => {
    if (status === 'pending') return t.statusPending;
    if (status === 'in_progress') return t.statusInProgress;
    if (status === 'answered') return t.statusAnswered;
    if (status === 'closed') return t.statusClosed;
    return status;
  };

  const unreadQueryIds = new Set(
    myNotifications
      .filter((n) => !n.is_read && typeof n.related_query_id === 'number')
      .map((n) => n.related_query_id as number)
  );

  const refreshSupportNotifications = async () => {
    if (!token) return;
    try {
      const notifications = await getMySupportNotifications(token);
      setMyNotifications(notifications);
      setSupportUnreadCount(notifications.filter((n) => !n.is_read).length);
    } catch (e) {
      console.error('Failed to refresh support notifications', e);
    }
  };

  const refreshChatReferencesVisibility = async (): Promise<boolean> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/faq/settings/chat-references/public?t=${Date.now()}`,
        { cache: 'no-store' }
      );
      if (!response.ok) return chatReferencesEnabledGlobal;
      const data = await response.json();
      if (typeof data?.enabled === 'boolean') {
        setChatReferencesEnabledGlobal(data.enabled);
        return data.enabled;
      }
    } catch (e) {
      console.error('Failed to refresh chat references visibility', e);
    }
    return chatReferencesEnabledGlobal;
  };

  const loadSupportPanelData = async () => {
    if (!token) return;
    setSupportPanelLoading(true);
    try {
      const [queries, notifications] = await Promise.all([
        getMySupportQueries(token),
        getMySupportNotifications(token),
      ]);
      setMySupportQueries(queries);
      setMyNotifications(notifications);
      setSupportUnreadCount(notifications.filter((n) => !n.is_read).length);
    } catch (e) {
      console.error('Failed to load support panel data', e);
    } finally {
      setSupportPanelLoading(false);
    }
  };

  const handleSubmitSupportQuery = async () => {
    if (!token || !supportMessage.trim()) return;
    setSupportLoading(true);
    setSupportError(null);
    setSupportSuccess(null);
    try {
      await createSupportQuery(token, {
        student_name: user?.full_name || undefined,
        phone: user?.phone || undefined,
        subject: `Support query from student`,
        message: supportMessage.trim(),
      });
      setSupportMessage('');
      setSupportSuccess(t.supportSent);
      await loadSupportPanelData();
    } catch (err) {
      setSupportError(err instanceof Error ? err.message : 'Failed to send support query');
    } finally {
      setSupportLoading(false);
    }
  };

  const markAllSupportNotificationsRead = async () => {
    if (!token) return;
    const unread = myNotifications.filter((n) => !n.is_read);
    if (unread.length === 0) return;
    try {
      await Promise.allSettled(unread.map((n) => markSupportNotificationRead(token, n.id)));
      setMyNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setSupportUnreadCount(0);
    } catch (e) {
      console.error('Failed to mark all support notifications read', e);
    }
  };

  // Redirect to login if not authenticated
  useEffect(() => {
    conversationIdRef.current = conversationId;
  }, [conversationId]);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.replace('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!showSupportPanel || !token) return;
    void (async () => {
      await loadSupportPanelData();
      await markAllSupportNotificationsRead();
    })();
  }, [showSupportPanel, token]);

  useEffect(() => {
    if (!token) return;
    void refreshSupportNotifications();
    void refreshChatReferencesVisibility();
    const stream = connectSupportNotificationStream(
      token,
      (event) => {
        if (event?.type === 'support_notification_created') {
          void refreshSupportNotifications();
          if (showSupportPanel) {
            void loadSupportPanelData();
          }
        }
      },
      () => {
        console.error('Support notification stream disconnected');
      }
    );
    return () => stream.close();
  }, [token]);

  useEffect(() => {
    const onFocus = () => {
      void refreshChatReferencesVisibility();
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [refreshChatReferencesVisibility]);

  const handleSendMessage = async (
    quickReply?: string,
    options?: { hideUserMessage?: boolean }
  ) => {
    const referencesEnabledForTurn = await refreshChatReferencesVisibility();
    const trimmed = (quickReply ?? inputValue).trim();
    const hideUserMessage = Boolean(options?.hideUserMessage);
    const pending = pendingClarification;

    if (isLoading) return;

    if (!pending && quickReply) {
      const intent = resolveGuidedIntent(quickReply);
      if (intent) {
        if (intent === 'college_shortlist') {
          // Let backend decide based on stored cutoff_profile/preferences_set.
          setGuidedIntent(null);
        } else {
        const guideMessage = t.guidedPrompts[intent];
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
          content: guideMessage,
          timestamp: new Date(),
        };
        setGuidedIntent(intent);
        setMessages((prev) => [...prev, userMessage, assistantMessage]);
        setInputValue('');
        return;
        }
      }
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
      assistantMessageId = `assistant-${turnId}`;
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        sources: [],
        referencesEnabled: referencesEnabledForTurn,
      };
      if (hideUserMessage) {
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        const userMessage: Message = {
          id: `user-${turnId}`,
          role: 'user',
          content: trimmed,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMessage, assistantMessage]);
      }
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
      const conversationIdForRequest =
        messages.length === 0 ? undefined : (conversationIdRef.current || undefined);
      await streamChatMessage(
        question,
        selectedModel,
        // onToken - append each token
        (token) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, cutoffInterpretationLoading: false, content: msg.content + token }
                : msg
            )
          );
        },
        // onSources - set sources
        (sources) => {
          const hasWeb = Array.isArray(sources) && sources.some((s) => {
            const dtype = String(s?.document_type || '').toLowerCase();
            const name = String(s?.file_name || '').toLowerCase();
            return dtype.includes('web') || name.startsWith('http://') || name.startsWith('https://');
          });
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === assistantMessageId 
                ? { ...msg, sources, sourceOrigin: hasWeb ? 'web' : 'kb' }
                : msg
            )
          );
        },
        // onMeta - set retrieval origin for badge rendering
        (meta) => {
          const refsEnabled = meta?.chat_references_enabled;
          const origin = meta?.source_origin;
          const cutoffInterpretationLoading = meta?.cutoff_interpretation_loading;
          const hasOrigin = origin === 'web' || origin === 'kb' || origin === 'none';
          const hasRefsFlag = typeof refsEnabled === 'boolean';
          const hasCutoffLoadingFlag = typeof cutoffInterpretationLoading === 'boolean';
          if (!hasOrigin && !hasRefsFlag && !hasCutoffLoadingFlag) return;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    ...(hasOrigin ? { sourceOrigin: origin } : {}),
                    ...(hasRefsFlag ? { referencesEnabled: refsEnabled } : {}),
                    ...(hasCutoffLoadingFlag ? { cutoffInterpretationLoading } : {}),
                  }
                : msg
            )
          );
        },
        // onDone - update model and conversation ID
        (filters, newConversationId) => {
          if (newConversationId && newConversationId !== conversationId) {
            setConversationId(newConversationId);
          }
          // Refresh sidebar once on completion.
          // Title updates already trigger a dedicated refresh in onTitle callback.
          setSidebarKey((prev) => prev + 1);
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
          if (normalizedReplies.some((reply) => Boolean(resolveGuidedIntent(reply)))) {
            setAllowStarterReplies(false);
          }
        },
        // onCutoffProfileForm
        (payload) => {
          const states = Array.isArray(payload?.states) ? payload.states : [];
          const categories = Array.isArray(payload?.categories) ? payload.categories : [];
          const subCategories = Array.isArray(payload?.sub_categories) ? payload.sub_categories : [];
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    content: '',
                    cutoffProfileForm: {
                      states,
                      categories,
                      subCategories,
                      selectedState: payload?.state || '',
                      selectedCategory: payload?.category || '',
                      selectedSubCategory: payload?.sub_category || '',
                    },
                  }
                : msg
            )
          );
        },
        // conversationId and userId
        conversationIdForRequest,
        user?.id,
        // onTitle - refresh sidebar when title is generated
        (title, convId) => {
          if (convId) {
            setSidebarKey(prev => prev + 1);
          }
        },
        selectedLanguage,
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

  const handleCutoffProfileSubmit = (payload: { state: string; category: string; subCategory?: string }) => {
    if (isLoading) return;
    const lines: string[] = [];
    if (payload.state && payload.state !== 'NOT_SURE') {
      lines.push(`Home state: ${payload.state}`);
    } else {
      lines.push('Home state: Not sure');
    }
    if (payload.category && payload.category !== 'NOT_SURE') {
      lines.push(`Category: ${payload.category}`);
    } else {
      lines.push('Category: Not sure');
    }
    if (payload.subCategory) {
      lines.push(`Sub-category: ${payload.subCategory}`);
    }
    lines.push('Proceed with college shortlist.');
    void handleSendMessage(lines.join('\n'), { hideUserMessage: true });
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
        sourceOrigin: (m.sources || []).some((s: any) => String(s?.document_type || '').toLowerCase().includes('web')) ? 'web' : ((m.sources || []).length > 0 ? 'kb' : undefined),
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
    conversationIdRef.current = null;
    setPendingClarification(null);
    setGuidedIntent(null);
    setAllowStarterReplies(true);
    setInputValue('');
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
    <div className="relative flex h-[100dvh] min-h-0 overflow-hidden overscroll-none">
      {/* Chat History Sidebar */}
      {isAuthenticated && token && (
        <ChatSidebar
          key={sidebarKey}
          token={token}
          currentConversationId={conversationId}
          onSelectConversation={handleSelectConversation}
          onNewChat={handleNewChat}
          onOpenSupport={() => {
            setSupportError(null);
            setSupportSuccess(null);
            setShowSupportModal(true);
          }}
          onOpenMyQueries={() => setShowSupportPanel(true)}
          onOpenProfile={() => router.push('/profile')}
          onLogout={() => logout()}
          isCollapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          language={selectedLanguage}
        />
      )}
      
      <main className="flex min-h-0 flex-col flex-1 min-w-0 bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 overflow-hidden">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-blue-100 dark:border-slate-700 px-2 md:px-6 py-2.5 shadow-sm sticky top-0 z-50 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 md:gap-3 min-w-0">
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="md:hidden p-2 rounded-lg border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-gray-300"
              aria-label="Open chats"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl shadow-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-base md:text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent truncate">
                Med Buddy
              </h1>
              <p className="hidden sm:block text-xs text-gray-500 dark:text-gray-400 font-medium">{t.poweredByHeader}</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 md:gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-full">
              <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
              <span className="text-xs font-medium text-green-700 dark:text-green-400">{t.officialNtaSource}</span>
            </div>

            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value as LanguageCode)}
              className="text-xs px-2 py-1.5 rounded-lg border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-700 dark:text-gray-200 max-w-[86px]"
              aria-label="Language"
            >
              <option value="en">{TRANSLATIONS.en.languageLabel}</option>
              <option value="hi">{TRANSLATIONS.hi.languageLabel}</option>
              <option value="mr">{TRANSLATIONS.mr.languageLabel}</option>
            </select>

            {!isAdminUser && (
              <>
                <button
                  onClick={() => {
                    setSupportError(null);
                    setSupportSuccess(null);
                    setShowSupportModal(true);
                  }}
                  className="hidden md:flex items-center gap-1.5 px-3 py-1.5 text-xs border border-blue-200 dark:border-blue-700 rounded-lg text-blue-700 dark:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/30"
                >
                  <MessageCircleQuestion className="w-3.5 h-3.5" />
                  <span>{t.contactSupport}</span>
                </button>

                <button
                  onClick={() => setShowSupportPanel(true)}
                  className="hidden md:flex items-center gap-1.5 px-3 py-1.5 text-xs border border-gray-200 dark:border-slate-600 rounded-lg text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-slate-700 relative"
                >
                  <Bell className="w-3.5 h-3.5" />
                  <span>{t.mySupportUpdates}</span>
                  {supportUnreadCount > 0 && (
                    <span className="absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] px-1 rounded-full bg-red-500 text-white text-[10px] leading-[18px] text-center font-semibold">
                      {supportUnreadCount > 99 ? '99+' : supportUnreadCount}
                    </span>
                  )}
                </button>
              </>
            )}
            
            {/* Theme Toggle */}
            <ThemeToggle language={selectedLanguage} />
            
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
                  <p className="text-xs font-semibold text-gray-800 group-hover:text-indigo-700 transition-colors">{t.dashboard}</p>
                  <p className="text-[10px] text-gray-400 group-hover:text-indigo-400 transition-colors -mt-0.5">{t.admin}</p>
                </div>
              </Link>
            )}
            
            {/* Auth buttons */}
            {!authLoading && (
              <>
                {isAuthenticated ? (
                  <div className="relative hidden md:block">
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
                            {t.myProfile}
                          </Link>
                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              logout();
                            }}
                            className="w-full flex items-center gap-3 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
                          >
                            <LogOut className="w-4 h-4" />
                            {t.signOut}
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="hidden md:flex items-center gap-2">
                    <Link
                      href="/login"
                      className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                    >
                      <LogIn className="w-4 h-4" />
                      <span className="hidden sm:inline">{t.signIn}</span>
                    </Link>
                    <Link
                      href="/register"
                      className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 rounded-lg transition-colors"
                    >
                      <UserPlus className="w-4 h-4" />
                      <span className="hidden sm:inline">{t.signUp}</span>
                    </Link>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
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
              {t.heroDescription}
            </p>
            
            {/* Trust Badges */}
            <div className="flex flex-wrap justify-center gap-3 mb-8">
              <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-full">
                <BookOpen className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-400">{t.officialNtaDocument}</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-full">
                <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-700 dark:text-green-400">{t.authenticInfo}</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-purple-50 dark:bg-purple-900/30 border border-purple-200 dark:border-purple-700 rounded-full">
                <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-medium text-purple-700 dark:text-purple-400">{t.aiPowered}</span>
              </div>
            </div>

            {/* Quick Questions */}
            <div className="w-full max-w-3xl">
              <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
                {t.startWith}
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <QuickQuestion
                  icon={<Calendar className="w-5 h-5" />}
                  onClick={() => void handleSendMessage(t.starter.neet_exam_guidance)}
                >
                  {t.starter.neet_exam_guidance}
                </QuickQuestion>
                <QuickQuestion
                  icon={<BookOpen className="w-5 h-5" />}
                  onClick={() => void handleSendMessage(t.starter.counselling_process)}
                >
                  {t.starter.counselling_process}
                </QuickQuestion>
                <QuickQuestion
                  icon={<HelpCircle className="w-5 h-5" />}
                  onClick={() => void handleSendMessage(t.starter.college_shortlist)}
                >
                  {t.starter.college_shortlist}
                </QuickQuestion>
                <QuickQuestion
                  icon={<FileCheck className="w-5 h-5" />}
                  onClick={() => void handleSendMessage(t.starter.college_fee_structure)}
                >
                  {t.starter.college_fee_structure}
                </QuickQuestion>
              </div>
            </div>

            {/* Note / disclaimer (empty-state footer) */}
            <p className="text-xs italic text-gray-400 dark:text-gray-500 mt-8 max-w-lg leading-relaxed">
              <span className="font-medium not-italic text-gray-500 dark:text-gray-400">{t.noteLabel}</span>
              {t.noteBody}
            </p>
          </div>
        ) : (
          <ChatWindow
            messages={displayedMessages}
            isLoading={isLoading}
            messagesEndRef={messagesEndRef}
            onSuggestedReply={handleSuggestedReply}
            onCutoffProfileSubmit={handleCutoffProfileSubmit}
            language={selectedLanguage}
            referencesEnabledGlobal={chatReferencesEnabledGlobal}
          />
        )}
      </div>

      {/* Input Area */}
      <div className="shrink-0 sticky bottom-0 z-30 bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-t border-blue-100 dark:border-slate-700 px-2 md:px-4 py-2.5 md:py-4 pb-[max(0.625rem,env(safe-area-inset-bottom))]">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-2 md:gap-3">
            <div className="flex-1 relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  pendingClarification
                    ? t.placeholderClarification
                    : t.placeholderDefault
                }
                className="w-full px-3 md:px-5 py-3 md:py-4 border border-gray-200 dark:border-slate-600 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-sm bg-white dark:bg-slate-700 text-gray-800 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500 text-sm md:text-base"
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={() => handleSendMessage()}
              disabled={!inputValue.trim() || isLoading}
              className="px-4 md:px-6 py-3 md:py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <LoadingSpinner />
                  <span className="hidden sm:inline">{t.searching}</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span className="hidden sm:inline">{t.ask}</span>
                </>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">
            {t.footerPowerBy} <span className="font-semibold text-blue-600 dark:text-blue-400">Get My University</span> • {t.footerEnter}
          </p>
        </div>
      </div>

      {showSupportModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowSupportModal(false)}
          />
          <div className="relative w-full max-w-lg bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-2xl p-5 mx-4 shadow-2xl">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">{t.supportModalTitle}</h3>
            <textarea
              value={supportMessage}
              onChange={(e) => setSupportMessage(e.target.value)}
              className="w-full min-h-[140px] p-3 rounded-xl border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-800 dark:text-white"
              placeholder={t.supportMessagePlaceholder}
            />
            {supportError && <p className="text-sm text-red-600 mt-2">{supportError}</p>}
            {supportSuccess && <p className="text-sm text-green-600 mt-2">{supportSuccess}</p>}
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => setShowSupportModal(false)}
                className="px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-slate-600 text-gray-700 dark:text-gray-200"
              >
                {t.close}
              </button>
              <button
                onClick={() => void handleSubmitSupportQuery()}
                disabled={supportLoading || !supportMessage.trim()}
                className="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white disabled:opacity-60"
              >
                {supportLoading ? t.loading : t.submitSupport}
              </button>
            </div>
          </div>
        </div>
      )}

      {showSupportPanel && (
        <div className="fixed inset-0 z-[90] flex justify-end">
          <div className="absolute inset-0 bg-black/40" onClick={() => setShowSupportPanel(false)} />
          <div className="relative h-full w-full max-w-xl bg-white dark:bg-slate-900 border-l border-gray-200 dark:border-slate-700 overflow-y-auto p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t.mySupportUpdates}</h3>
              <button
                onClick={() => setShowSupportPanel(false)}
                className="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-300 dark:border-slate-500 bg-white dark:bg-slate-800 text-gray-700 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-slate-700 shadow-sm"
              >
                {t.close}
              </button>
            </div>

            <section>
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">{t.supportHistoryTitle}</h4>
              {supportPanelLoading ? (
                <p className="text-sm text-gray-500">{t.loading}</p>
              ) : mySupportQueries.length === 0 ? (
                <p className="text-sm text-gray-500">{t.noSupportQueries}</p>
              ) : (
                <div className="space-y-3">
                  {mySupportQueries.map((q) => {
                    const latestReply = q.replies && q.replies.length > 0 ? q.replies[q.replies.length - 1] : null;
                    return (
                      <div key={q.id} className="p-3 rounded-lg border border-gray-200 dark:border-slate-700">
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-sm font-medium text-gray-800 dark:text-gray-100">{q.subject}</p>
                          <div className="flex items-center gap-2">
                            {unreadQueryIds.has(q.id) && (
                              <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
                                {t.unreadLabel}
                              </span>
                            )}
                            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300">
                              {supportStatusLabel(q.status)}
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-300 mt-1 whitespace-pre-wrap">{q.message}</p>
                        {latestReply && (
                          <div className="mt-2 p-2 rounded bg-blue-50 dark:bg-blue-900/20">
                            <p className="text-xs font-medium text-blue-700 dark:text-blue-300">{t.latestReplyLabel}</p>
                            <p className="text-sm text-blue-900 dark:text-blue-100 whitespace-pre-wrap">{latestReply.reply_text}</p>
                          </div>
                        )}
                        <p className="text-xs text-gray-400 mt-2">{new Date(q.created_at).toLocaleString()}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        </div>
      )}
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
