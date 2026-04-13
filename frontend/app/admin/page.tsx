'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import ThemeToggle from '@/components/ThemeToggle';
import { 
  Upload, FileText, Trash2, RefreshCw, Database, Users, Settings,
  CheckCircle, AlertCircle, ArrowLeft, Search, ChevronLeft, ChevronRight,
  ShieldAlert, Loader2, Edit2, X, Key, BarChart3, TrendingUp, Shield, 
  UserX, Layers, MessageSquare, Check, XCircle, Eye, Plus, FileUp, Sparkles,
  Pause, Play, Globe, Download
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
interface DashboardStats {
  total_users: number;
  active_users: number;
  verified_users: number;
  admin_users: number;
  total_documents: number;
  total_vectors: number;
  users_by_role: Record<string, number>;
  recent_signups: number;
}

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  phone: string | null;
  age: number | null;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  target_exams: string[];
  created_at: string;
  last_login_at: string | null;
}

interface Document {
  id: number;
  file_id: string;
  filename: string;
  original_filename: string;
  state: string;
  document_type: string;
  category: string;
  year: string;
  description: string | null;
  total_pages: number;
  total_vectors: number;
  file_size_kb: number;
  is_active: boolean;
  index_status: string;
  indexed_at: string;
  storage_url: string | null;
  uploaded_by: number | null;
}

interface MetadataOptions {
  states: string[];
  document_types: { value: string; label: string }[];
  categories: { value: string; label: string }[];
  years: string[];
}

interface FAQ {
  id: number;
  question: string;
  original_answer: string;
  modified_answer: string | null;
  detected_state: string | null;
  detected_exam: string | null;
  detected_category: string | null;
  status: 'pending' | 'approved' | 'rejected' | 'modified';
  occurrence_count: number;
  faq_vector_id: string | null;
  created_at: string;
  reviewed_at: string | null;
}

interface FAQStats {
  total: number;
  pending_review: number;
  approved: number;
  rejected: number;
}

type TabType = 'overview' | 'users' | 'documents' | 'faqs';

export default function AdminPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  
  // Tab state
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  
  // General state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Per-tab loading states for instant feedback
  const [usersLoading, setUsersLoading] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [faqsLoading, setFaqsLoading] = useState(false);
  
  // Dashboard stats
  const [stats, setStats] = useState<DashboardStats | null>(null);
  
  // Users state
  const [users, setUsers] = useState<User[]>([]);
  const [usersTotal, setUsersTotal] = useState(0);
  const [usersPage, setUsersPage] = useState(1);
  const [usersTotalPages, setUsersTotalPages] = useState(1);
  const [usersSearch, setUsersSearch] = useState('');
  const [usersRoleFilter, setUsersRoleFilter] = useState('');
  const [usersActiveFilter, setUsersActiveFilter] = useState<string>('');
  const [editingUser, setEditingUser] = useState<User | null>(null);
  
  // Documents state
  const [documents, setDocuments] = useState<Document[]>([]);
  const [docsTotal, setDocsTotal] = useState(0);
  const [docsPage, setDocsPage] = useState(1);
  const [docsTotalPages, setDocsTotalPages] = useState(1);
  const [docsSearch, setDocsSearch] = useState('');
  const [docsStateFilter, setDocsStateFilter] = useState('');
  const [docsDocTypeFilter, setDocsDocTypeFilter] = useState('');
  
  // Upload state
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [metadataOptions, setMetadataOptions] = useState<MetadataOptions | null>(null);
  const [uploadForm, setUploadForm] = useState({
    state: 'All-India',
    document_type: 'nta_bulletin',
    category: 'general',
    year: '2026',
    description: ''
  });
  
  // FAQ state
  const [faqs, setFaqs] = useState<FAQ[]>([]);
  const [faqsTotal, setFaqsTotal] = useState(0);
  const [faqsPage, setFaqsPage] = useState(1);
  const [faqsTotalPages, setFaqsTotalPages] = useState(1);
  const [faqsSearch, setFaqsSearch] = useState('');
  const [faqsStatusFilter, setFaqsStatusFilter] = useState('');
  const [faqStats, setFaqStats] = useState<FAQStats | null>(null);
  const [editingFaq, setEditingFaq] = useState<FAQ | null>(null);
  const [reviewingFaq, setReviewingFaq] = useState<FAQ | null>(null);
  const [showCreateFaq, setShowCreateFaq] = useState(false);
  const [showBulkUpload, setShowBulkUpload] = useState(false);
  const [reviewingFaqAction, setReviewingFaqAction] = useState<{ id: number; action: string } | null>(null);
  
  // Auto-learning state
  const [autoLearningEnabled, setAutoLearningEnabled] = useState<boolean>(true);
  const [autoLearningLoading, setAutoLearningLoading] = useState<boolean>(false);
  const [autoLearningUpdatedAt, setAutoLearningUpdatedAt] = useState<string | null>(null);
  const [webFallbackEnabled, setWebFallbackEnabled] = useState<boolean>(false);
  const [webFallbackLoading, setWebFallbackLoading] = useState<boolean>(false);
  const [webFallbackUpdatedAt, setWebFallbackUpdatedAt] = useState<string | null>(null);
  
  // Confirmation modal state
  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    details?: string[];
    type: 'danger' | 'warning' | 'info';
    onConfirm: () => void;
    confirmText?: string;
    isLoading?: boolean;
  } | null>(null);
  
  // Get auth token
  const getAuthToken = () => {
    const tokens = localStorage.getItem('auth_tokens');
    if (tokens) {
      return JSON.parse(tokens).access_token;
    }
    return null;
  };
  
  // Auth check
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    } else if (!authLoading && isAuthenticated && user?.role !== 'admin' && user?.role !== 'super_admin') {
      router.push('/');
    }
  }, [authLoading, isAuthenticated, user, router]);

  // Fetch dashboard stats
  const fetchStats = useCallback(async () => {
    try {
      const token = getAuthToken();
      // Use combined overview endpoint for faster initial load (single API call)
      const response = await fetch(`${API_BASE_URL}/admin/dashboard/overview`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data.stats);
        // Pre-populate users and documents from overview response
        if (data.recent_users && users.length === 0) {
          // Don't override if we already have full user list
        }
        if (data.recent_documents && documents.length === 0) {
          // Don't override if we already have full document list
        }
        return data; // Return for parallel loading
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      // Fallback to stats-only endpoint
      try {
        const token = getAuthToken();
        const response = await fetch(`${API_BASE_URL}/admin/dashboard/stats`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        }
      } catch (fallbackErr) {
        console.error('Fallback stats fetch also failed:', fallbackErr);
      }
    }
  }, [users.length, documents.length]);

  // Fetch users
  const fetchUsers = useCallback(async (showLoader = true) => {
    if (showLoader) setUsersLoading(true);
    try {
      const token = getAuthToken();
      const params = new URLSearchParams({
        page: usersPage.toString(),
        page_size: '10',
        ...(usersSearch && { search: usersSearch }),
        ...(usersRoleFilter && { role: usersRoleFilter }),
        ...(usersActiveFilter !== '' && { is_active: usersActiveFilter })
      });
      
      const response = await fetch(`${API_BASE_URL}/admin/users?${params}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setUsers(data.users);
        setUsersTotal(data.total);
        setUsersTotalPages(data.total_pages);
      }
    } catch (err) {
      console.error('Failed to fetch users:', err);
    } finally {
      setUsersLoading(false);
    }
  }, [usersPage, usersSearch, usersRoleFilter, usersActiveFilter]);

  // Fetch documents
  const fetchDocuments = useCallback(async (showLoader = true) => {
    if (showLoader) setDocsLoading(true);
    try {
      const token = getAuthToken();
      const params = new URLSearchParams({
        page: docsPage.toString(),
        page_size: '10',
        ...(docsSearch && { search: docsSearch }),
        ...(docsStateFilter && { state: docsStateFilter }),
        ...(docsDocTypeFilter && { document_type: docsDocTypeFilter }),
      });
      
      const response = await fetch(`${API_BASE_URL}/admin/documents?${params}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents);
        setDocsTotal(data.total);
        setDocsTotalPages(data.total_pages);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setDocsLoading(false);
    }
  }, [docsPage, docsSearch, docsStateFilter, docsDocTypeFilter]);

  // Fetch FAQs
  const fetchFaqs = useCallback(async (showLoader = true) => {
    if (showLoader) setFaqsLoading(true);
    try {
      const token = getAuthToken();
      const params = new URLSearchParams({
        page: faqsPage.toString(),
        page_size: '10',
        ...(faqsSearch && { search: faqsSearch }),
        ...(faqsStatusFilter && { status_filter: faqsStatusFilter })
      });
      
      const response = await fetch(`${API_BASE_URL}/faq/list?${params}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setFaqs(data.faqs);
        setFaqsTotal(data.total);
        setFaqsTotalPages(Math.ceil(data.total / data.page_size));
      }
    } catch (err) {
      console.error('Failed to fetch FAQs:', err);
    } finally {
      setFaqsLoading(false);
    }
  }, [faqsPage, faqsSearch, faqsStatusFilter]);

  // Fetch FAQ stats
  const fetchFaqStats = useCallback(async () => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/stats/overview`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setFaqStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch FAQ stats:', err);
    }
  }, []);

  // Fetch auto-learning status
  const fetchAutoLearningStatus = useCallback(async () => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/settings/auto-learning`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setAutoLearningEnabled(data.enabled);
        setAutoLearningUpdatedAt(data.updated_at);
      }
    } catch (err) {
      console.error('Failed to fetch auto-learning status:', err);
    }
  }, []);

  // Toggle auto-learning
  const toggleAutoLearning = async () => {
    setAutoLearningLoading(true);
    try {
      const token = getAuthToken();
      const newStatus = !autoLearningEnabled;
      const response = await fetch(`${API_BASE_URL}/faq/settings/auto-learning?enable=${newStatus}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setAutoLearningEnabled(data.enabled);
        setAutoLearningUpdatedAt(data.updated_at);
        setSuccess(data.message);
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to update auto-learning setting');
      }
    } catch (err) {
      setError('Failed to update auto-learning setting');
    } finally {
      setAutoLearningLoading(false);
    }
  };

  // Fetch web-search fallback status
  const fetchWebFallbackStatus = useCallback(async () => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/settings/web-search-fallback`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setWebFallbackEnabled(data.enabled);
        setWebFallbackUpdatedAt(data.updated_at);
      }
    } catch (err) {
      console.error('Failed to fetch web fallback status:', err);
    }
  }, []);

  // Toggle web-search fallback
  const toggleWebFallback = async () => {
    setWebFallbackLoading(true);
    try {
      const token = getAuthToken();
      const newStatus = !webFallbackEnabled;
      const response = await fetch(`${API_BASE_URL}/faq/settings/web-search-fallback?enable=${newStatus}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setWebFallbackEnabled(data.enabled);
        setWebFallbackUpdatedAt(data.updated_at);
        setSuccess(data.message);
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to update web fallback setting');
      }
    } catch (err) {
      setError('Failed to update web fallback setting');
    } finally {
      setWebFallbackLoading(false);
    }
  };

  // Fetch metadata options
  const fetchMetadataOptions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/admin/metadata-options`);
      if (response.ok) {
        const data = await response.json();
        setMetadataOptions(data);
      }
    } catch (err) {
      console.error('Failed to fetch metadata options:', err);
    }
  }, []);

  // Initial load - use combined overview endpoint first for fast stats
  useEffect(() => {
    const loadAllData = async () => {
      setLoading(true);
      
      // Step 1: Fetch stats first (using combined endpoint for single round trip)
      await fetchStats();
      setLoading(false); // Show stats immediately
      
      // Step 2: Load remaining data in background (user can already see stats)
      await Promise.all([
        fetchMetadataOptions(),
        fetchUsers(false),      // false = don't show loader (initial load)
        fetchDocuments(false),
        fetchFaqs(false),
        fetchFaqStats(),
        fetchAutoLearningStatus(),
        fetchWebFallbackStatus()
      ]);
      setInitialLoadDone(true);
    };
    if (isAuthenticated && (user?.role === 'admin' || user?.role === 'super_admin')) {
      loadAllData();
    }
  }, [isAuthenticated, user, fetchStats, fetchMetadataOptions, fetchUsers, fetchDocuments, fetchFaqs, fetchFaqStats, fetchAutoLearningStatus, fetchWebFallbackStatus]);

  // Refetch tab data when filters/pagination change (not on initial load since data is pre-loaded)
  const [initialLoadDone, setInitialLoadDone] = useState(false);
  
  useEffect(() => {
    if (initialLoadDone && activeTab === 'users') {
      fetchUsers();
    }
  }, [usersPage, usersSearch, usersRoleFilter, usersActiveFilter, initialLoadDone]);

  useEffect(() => {
    if (initialLoadDone && activeTab === 'documents') {
      fetchDocuments();
    }
  }, [docsPage, docsSearch, docsStateFilter, docsDocTypeFilter, initialLoadDone, fetchDocuments]);

  useEffect(() => {
    if (initialLoadDone && activeTab === 'faqs') {
      fetchFaqs();
    }
  }, [faqsPage, faqsSearch, faqsStatusFilter, initialLoadDone]);

  // User actions
  const handleUpdateUser = async (userId: number, updates: Partial<User>) => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/admin/users/${userId}`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updates)
      });
      
      if (response.ok) {
        setSuccess('User updated successfully');
        setEditingUser(null);
        fetchUsers();
        fetchStats();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to update user');
      }
    } catch (err) {
      setError('Failed to update user');
    }
  };

  const handleDeleteUser = async (userId: number, username: string) => {
    if (!confirm(`Are you sure you want to deactivate user "${username}"?`)) return;
    
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/admin/users/${userId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        setSuccess('User deactivated successfully');
        fetchUsers();
        fetchStats();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to deactivate user');
      }
    } catch (err) {
      setError('Failed to deactivate user');
    }
  };

  const handleResetPassword = async (userId: number, username: string) => {
    if (!confirm(`Reset password for "${username}"?`)) return;
    
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/admin/users/${userId}/reset-password`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setSuccess(`Password reset! Temp: ${data.temporary_password}`);
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to reset password');
      }
    } catch (err) {
      setError('Failed to reset password');
    }
  };

  // Document actions
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files are supported');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('state', uploadForm.state);
    formData.append('document_type', uploadForm.document_type);
    formData.append('category', uploadForm.category);
    formData.append('year', uploadForm.year);
    formData.append('description', uploadForm.description);

    try {
      const response = await fetch(`${API_BASE_URL}/admin/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok && result.success) {
        setSuccess(`Indexed ${result.pages_indexed} pages!`);
        setSelectedFile(null);
        const fileInput = document.getElementById('file-upload') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
        fetchDocuments();
        fetchStats();
      } else {
        setError(result.detail || 'Upload failed');
      }
    } catch (err) {
      setError('Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  // Track which documents are being reindexed
  const [reindexingDocs, setReindexingDocs] = useState<Set<number>>(new Set());

  const handleDeleteDocument = (docId: number, filename: string) => {
    setConfirmModal({
      isOpen: true,
      title: 'Delete Document',
      message: `Are you sure you want to delete "${filename}"?`,
      details: [
        'This will permanently remove the document from storage',
        'All indexed vectors will be deleted from Pinecone',
        'This action cannot be undone'
      ],
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: async () => {
        setConfirmModal(prev => prev ? { ...prev, isLoading: true } : null);
        try {
          const token = getAuthToken();
          const response = await fetch(`${API_BASE_URL}/admin/documents/${docId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}` }
          });
          
          if (response.ok) {
            setSuccess('Document deleted successfully');
            fetchDocuments();
            fetchStats();
          } else {
            const err = await response.json();
            setError(err.detail || 'Failed to delete');
          }
        } catch (err) {
          setError('Failed to delete document');
        } finally {
          setConfirmModal(null);
        }
      }
    });
  };

  const handleReindexDocument = (docId: number, filename: string) => {
    setConfirmModal({
      isOpen: true,
      title: 'Reindex Document',
      message: `Reindex "${filename}" with updated chunk classification?`,
      details: [
        'Download the PDF from storage',
        'Re-classify all chunks with proper categories',
        'Update vectors in Pinecone with new metadata',
        'This may take 1-2 minutes for large documents'
      ],
      type: 'info',
      confirmText: 'Reindex',
      onConfirm: async () => {
        setConfirmModal(null);
        setReindexingDocs(prev => new Set(prev).add(docId));
        
        try {
          const token = getAuthToken();
          const response = await fetch(`${API_BASE_URL}/admin/documents/${docId}/reindex`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
          });
          
          if (response.ok) {
            const result = await response.json();
            setSuccess(`Reindexed "${filename}" with ${result.document.total_vectors} vectors`);
            fetchDocuments();
            fetchStats();
          } else {
            const err = await response.json();
            setError(err.detail || 'Failed to reindex');
          }
        } catch (err) {
          setError('Failed to reindex document');
        } finally {
          setReindexingDocs(prev => {
            const next = new Set(prev);
            next.delete(docId);
            return next;
          });
        }
      }
    });
  };

  const handleViewDocument = (doc: Document) => {
    const token = getAuthToken();
    if (!token) {
      setError('Authentication required to view documents');
      return;
    }

    (async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/documents/${doc.id}/file`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) {
          throw new Error('Failed to fetch document');
        }
        const blob = await response.blob();
        const objectUrl = window.URL.createObjectURL(blob);
        window.open(objectUrl, '_blank', 'noopener,noreferrer');
        setTimeout(() => window.URL.revokeObjectURL(objectUrl), 60000);
      } catch (err) {
        setError('Unable to open document');
      }
    })();
  };

  const handleDownloadDocument = async (doc: Document) => {
    const token = getAuthToken();
    if (!token) {
      setError('Authentication required to download documents');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/admin/documents/${doc.id}/file?download=true`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!response.ok) {
        throw new Error('Download failed');
      }

      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = doc.original_filename || 'document.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setError('Unable to download document');
    }
  };

  // FAQ actions
  const handleReviewFaq = async (faqId: number, action: 'approve' | 'reject' | 'modify', modifiedAnswer?: string, reviewNotes?: string) => {
    setReviewingFaqAction({ id: faqId, action });
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/${faqId}/review`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action, modified_answer: modifiedAnswer, review_notes: reviewNotes })
      });
      
      if (response.ok) {
        setSuccess(`FAQ ${action === 'approve' ? 'approved & vectorized' : action + 'd'} successfully`);
        setReviewingFaq(null);
        fetchFaqs();
        fetchFaqStats();
      } else {
        const err = await response.json();
        setError(err.detail || `Failed to ${action} FAQ`);
      }
    } catch (err) {
      setError(`Failed to ${action} FAQ`);
    } finally {
      setReviewingFaqAction(null);
    }
  };

  const handleCreateFaq = async (question: string, answer: string, state?: string, category?: string) => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/create`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question, answer, state, category, exam: 'NEET' })
      });
      
      if (response.ok) {
        setSuccess('FAQ created successfully');
        setShowCreateFaq(false);
        fetchFaqs();
        fetchFaqStats();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to create FAQ');
      }
    } catch (err) {
      setError('Failed to create FAQ');
    }
  };

  const handleDeleteFaq = async (faqId: number) => {
    if (!confirm('Delete this FAQ?')) return;
    
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/${faqId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        setSuccess('FAQ deleted');
        fetchFaqs();
        fetchFaqStats();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to delete FAQ');
      }
    } catch (err) {
      setError('Failed to delete FAQ');
    }
  };

  const handleBulkUploadFaqs = async (faqs: { question: string; answer: string; state?: string; category?: string }[], autoApprove: boolean) => {
    try {
      const token = getAuthToken();
      const response = await fetch(`${API_BASE_URL}/faq/bulk-upload?auto_approve=${autoApprove}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(faqs)
      });
      
      if (response.ok) {
        const data = await response.json();
        setSuccess(`Uploaded ${data.created} FAQs${autoApprove ? ' (auto-approved)' : ''}`);
        setShowBulkUpload(false);
        fetchFaqs();
        fetchFaqStats();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to bulk upload FAQs');
      }
    } catch (err) {
      setError('Failed to bulk upload FAQs');
    }
  };

  // Clear messages
  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => {
        setError(null);
        setSuccess(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, success]);

  // Loading state
  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  // Access denied
  if (!isAuthenticated || (user?.role !== 'admin' && user?.role !== 'super_admin')) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="bg-slate-800 rounded-2xl p-8 max-w-md text-center border border-slate-700">
          <ShieldAlert className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Access Denied</h2>
          <p className="text-slate-400 mb-6">Admin access required.</p>
          <Link href="/" className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <ArrowLeft className="w-4 h-4" /> Go Back
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-slate-900">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-lg border-b border-gray-200 dark:border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="flex items-center gap-2 text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div className="flex items-center gap-3">
                <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-xl">
                  <Settings className="w-5 h-5 text-white" />
                </div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">Admin Dashboard</h1>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500 dark:text-slate-400">
                <span className="text-blue-600 dark:text-blue-400">{user?.username}</span>
              </span>
              <ThemeToggle />
              <button
                onClick={() => {
                  fetchStats();
                  if (activeTab === 'users') fetchUsers();
                  if (activeTab === 'documents') fetchDocuments();
                  if (activeTab === 'faqs') { fetchFaqs(); fetchFaqStats(); fetchAutoLearningStatus(); fetchWebFallbackStatus(); }
                }}
                className="p-2 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg"
              >
                <RefreshCw className={`w-4 h-4 text-gray-600 dark:text-slate-300 ${
                  loading || usersLoading || docsLoading || faqsLoading ? 'animate-spin' : ''
                }`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Alerts */}
      {(error || success) && (
        <div className="max-w-7xl mx-auto px-6 pt-4">
          {error && (
            <div className="bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-600 dark:text-red-400 flex-1">{error}</p>
              <button onClick={() => setError(null)}><X className="w-4 h-4 text-red-500 dark:text-red-400" /></button>
            </div>
          )}
          {success && (
            <div className="bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/20 rounded-lg p-4 flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <p className="text-green-600 dark:text-green-400 flex-1">{success}</p>
              <button onClick={() => setSuccess(null)}><X className="w-4 h-4 text-green-500 dark:text-green-400" /></button>
            </div>
          )}
        </div>
      )}

      {/* Confirmation Modal */}
      {confirmModal?.isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => !confirmModal.isLoading && setConfirmModal(null)}
          />
          
          {/* Modal */}
          <div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Header with icon */}
            <div className={`px-6 py-5 ${
              confirmModal.type === 'danger' ? 'bg-gradient-to-r from-red-500/10 to-red-600/5' :
              confirmModal.type === 'warning' ? 'bg-gradient-to-r from-yellow-500/10 to-orange-500/5' :
              'bg-gradient-to-r from-blue-500/10 to-indigo-500/5'
            }`}>
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl ${
                  confirmModal.type === 'danger' ? 'bg-red-500/20' :
                  confirmModal.type === 'warning' ? 'bg-yellow-500/20' :
                  'bg-blue-500/20'
                }`}>
                  {confirmModal.type === 'danger' ? (
                    <Trash2 className="w-6 h-6 text-red-500" />
                  ) : confirmModal.type === 'warning' ? (
                    <AlertCircle className="w-6 h-6 text-yellow-500" />
                  ) : (
                    <RefreshCw className="w-6 h-6 text-blue-500" />
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {confirmModal.title}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-slate-400 mt-0.5">
                    {confirmModal.message}
                  </p>
                </div>
              </div>
            </div>
            
            {/* Details */}
            {confirmModal.details && confirmModal.details.length > 0 && (
              <div className="px-6 py-4 border-t border-gray-100 dark:border-slate-700">
                <p className="text-xs font-medium text-gray-500 dark:text-slate-500 uppercase tracking-wider mb-3">
                  What will happen
                </p>
                <ul className="space-y-2">
                  {confirmModal.details.map((detail, idx) => (
                    <li key={idx} className="flex items-start gap-2.5 text-sm text-gray-600 dark:text-slate-300">
                      <div className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                        confirmModal.type === 'danger' ? 'bg-red-400' :
                        confirmModal.type === 'warning' ? 'bg-yellow-400' :
                        'bg-blue-400'
                      }`} />
                      {detail}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Actions */}
            <div className="px-6 py-4 bg-gray-50 dark:bg-slate-800/50 border-t border-gray-100 dark:border-slate-700 flex gap-3 justify-end">
              <button
                onClick={() => setConfirmModal(null)}
                disabled={confirmModal.isLoading}
                className="px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-xl hover:bg-gray-50 dark:hover:bg-slate-600 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmModal.onConfirm}
                disabled={confirmModal.isLoading}
                className={`px-5 py-2.5 text-sm font-medium text-white rounded-xl disabled:opacity-50 transition-all flex items-center gap-2 ${
                  confirmModal.type === 'danger' 
                    ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 shadow-lg shadow-red-500/25' 
                    : confirmModal.type === 'warning'
                    ? 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 shadow-lg shadow-yellow-500/25'
                    : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 shadow-lg shadow-blue-500/25'
                }`}
              >
                {confirmModal.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  confirmModal.confirmText || 'Confirm'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-6 pt-6">
        <div className="flex gap-1 bg-gray-100 dark:bg-slate-800/50 p-1 rounded-xl w-fit">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'users', label: 'Users', icon: Users },
            { id: 'documents', label: 'Documents', icon: FileText },
            { id: 'faqs', label: 'FAQs', icon: MessageSquare, badge: faqStats?.pending_review }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-slate-700/50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.badge && tab.badge > 0 && (
                <span className="px-1.5 py-0.5 text-xs bg-orange-500 text-white rounded-full">{tab.badge}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {loading && !stats ? (
                // Skeleton loading for stat cards
                <>
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="bg-white dark:bg-slate-800/50 rounded-2xl border border-gray-200 dark:border-slate-700 p-6 animate-pulse">
                      <div className="flex items-center justify-between mb-4">
                        <div className="h-4 bg-gray-200 dark:bg-slate-700 rounded w-20"></div>
                        <div className="w-10 h-10 bg-gray-200 dark:bg-slate-700 rounded-xl"></div>
                      </div>
                      <div className="h-8 bg-gray-200 dark:bg-slate-700 rounded w-16 mb-2"></div>
                      <div className="h-3 bg-gray-200 dark:bg-slate-700 rounded w-24"></div>
                    </div>
                  ))}
                </>
              ) : (
                <>
                  <StatCard title="Total Users" value={stats?.total_users || 0} icon={Users} color="blue" subtitle={`${stats?.recent_signups || 0} this week`} />
                  <StatCard title="Active Users" value={stats?.active_users || 0} icon={CheckCircle} color="green" subtitle={`${stats?.verified_users || 0} verified`} />
                  <StatCard title="Documents" value={stats?.total_documents || 0} icon={FileText} color="purple" subtitle="Indexed" />
                  <StatCard title="Vectors" value={stats?.total_vectors?.toLocaleString() || '0'} icon={Database} color="orange" subtitle="Embeddings" />
                </>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-slate-800/50 rounded-2xl border border-gray-200 dark:border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-blue-500 dark:text-blue-400" /> Users by Role
                </h3>
                <div className="space-y-4">
                  {stats?.users_by_role && Object.entries(stats.users_by_role).map(([role, count]) => (
                    <div key={role} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${
                          role === 'super_admin' ? 'bg-red-500' : role === 'admin' ? 'bg-orange-500' : 'bg-blue-500'
                        }`} />
                        <span className="text-gray-600 dark:text-slate-300 capitalize">{role.replace('_', ' ')}</span>
                      </div>
                      <span className="text-gray-900 dark:text-white font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white dark:bg-slate-800/50 rounded-2xl border border-gray-200 dark:border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500 dark:text-green-400" /> Quick Actions
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <button onClick={() => setActiveTab('users')} className="flex items-center gap-3 p-4 bg-gray-100 dark:bg-slate-700/50 hover:bg-gray-200 dark:hover:bg-slate-700 rounded-xl">
                    <Users className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                    <span className="text-gray-600 dark:text-slate-300">Manage Users</span>
                  </button>
                  <button onClick={() => setActiveTab('documents')} className="flex items-center gap-3 p-4 bg-gray-100 dark:bg-slate-700/50 hover:bg-gray-200 dark:hover:bg-slate-700 rounded-xl">
                    <Upload className="w-5 h-5 text-purple-500 dark:text-purple-400" />
                    <span className="text-gray-600 dark:text-slate-300">Upload Doc</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="space-y-6">
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-4">
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px] relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search users..."
                    value={usersSearch}
                    onChange={(e) => { setUsersSearch(e.target.value); setUsersPage(1); }}
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
                <select value={usersRoleFilter} onChange={(e) => { setUsersRoleFilter(e.target.value); setUsersPage(1); }}
                  className="px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                  <option value="">All Roles</option>
                  <option value="student">Student</option>
                  <option value="admin">Admin</option>
                  <option value="super_admin">Super Admin</option>
                </select>
                <select value={usersActiveFilter} onChange={(e) => { setUsersActiveFilter(e.target.value); setUsersPage(1); }}
                  className="px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                  <option value="">All Status</option>
                  <option value="true">Active</option>
                  <option value="false">Inactive</option>
                </select>
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left px-6 py-4 text-sm font-medium text-slate-400">User</th>
                      <th className="text-left px-6 py-4 text-sm font-medium text-slate-400">Contact</th>
                      <th className="text-left px-6 py-4 text-sm font-medium text-slate-400">Role</th>
                      <th className="text-left px-6 py-4 text-sm font-medium text-slate-400">Status</th>
                      <th className="text-left px-6 py-4 text-sm font-medium text-slate-400">Joined</th>
                      <th className="text-right px-6 py-4 text-sm font-medium text-slate-400">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700">
                    {usersLoading && users.length === 0 ? (
                      // Skeleton loading rows
                      <>
                        {[1, 2, 3, 4, 5].map((i) => (
                          <tr key={i} className="animate-pulse">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-slate-700 rounded-full"></div>
                                <div>
                                  <div className="h-4 bg-slate-700 rounded w-24 mb-1"></div>
                                  <div className="h-3 bg-slate-700 rounded w-16"></div>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="h-4 bg-slate-700 rounded w-32 mb-1"></div>
                              <div className="h-3 bg-slate-700 rounded w-20"></div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="h-6 bg-slate-700 rounded-full w-16"></div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="h-4 bg-slate-700 rounded w-16"></div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="h-4 bg-slate-700 rounded w-20"></div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex justify-end gap-2">
                                <div className="w-8 h-8 bg-slate-700 rounded-lg"></div>
                                <div className="w-8 h-8 bg-slate-700 rounded-lg"></div>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </>
                    ) : (
                    users.map((u) => (
                      <tr key={u.id} className="hover:bg-slate-700/30">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                              <span className="text-white font-medium">{u.full_name[0]}</span>
                            </div>
                            <div>
                              <p className="text-white font-medium">{u.full_name}</p>
                              <p className="text-slate-400 text-sm">@{u.username}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <p className="text-slate-300 text-sm">{u.email}</p>
                          <p className="text-slate-500 text-sm">{u.phone || '-'}</p>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                            u.role === 'super_admin' ? 'bg-red-500/10 text-red-400' :
                            u.role === 'admin' ? 'bg-orange-500/10 text-orange-400' : 'bg-blue-500/10 text-blue-400'
                          }`}>{u.role.replace('_', ' ')}</span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${u.is_active ? 'bg-green-500' : 'bg-red-500'}`} />
                            <span className={`text-sm ${u.is_active ? 'text-green-400' : 'text-red-400'}`}>
                              {u.is_active ? 'Active' : 'Inactive'}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-slate-400 text-sm">
                          {new Date(u.created_at).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end gap-2">
                            <button onClick={() => setEditingUser(u)} className="p-2 hover:bg-slate-600 rounded-lg" title="Edit">
                              <Edit2 className="w-4 h-4 text-slate-400" />
                            </button>
                            <button onClick={() => handleResetPassword(u.id, u.username)} className="p-2 hover:bg-slate-600 rounded-lg" title="Reset Password">
                              <Key className="w-4 h-4 text-slate-400" />
                            </button>
                            {u.role !== 'super_admin' && (
                              <button onClick={() => handleDeleteUser(u.id, u.username)} className="p-2 hover:bg-red-500/10 rounded-lg" title="Deactivate">
                                <UserX className="w-4 h-4 text-red-400" />
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                    )}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between px-6 py-4 border-t border-slate-700">
                <p className="text-sm text-slate-400">
                  {((usersPage - 1) * 10) + 1} - {Math.min(usersPage * 10, usersTotal)} of {usersTotal}
                </p>
                <div className="flex gap-2">
                  <button onClick={() => setUsersPage(p => Math.max(1, p - 1))} disabled={usersPage === 1}
                    className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                    <ChevronLeft className="w-4 h-4 text-white" />
                  </button>
                  <button onClick={() => setUsersPage(p => Math.min(usersTotalPages, p + 1))} disabled={usersPage === usersTotalPages}
                    className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                    <ChevronRight className="w-4 h-4 text-white" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="space-y-6">
            {/* Upload */}
            <div className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 rounded-2xl border border-blue-500/20 p-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-blue-400" /> Upload Document
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <label className="block cursor-pointer min-w-0">
                  <div className="border-2 border-dashed border-slate-600 hover:border-blue-500 rounded-xl p-6 text-center min-w-0 max-w-full overflow-hidden">
                    <input id="file-upload" type="file" accept=".pdf" onChange={handleFileChange} className="hidden" />
                    <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2 shrink-0" />
                    <p
                      className={`text-slate-300 text-sm w-full min-w-0 ${selectedFile ? 'truncate' : ''}`}
                      title={selectedFile ? selectedFile.name : undefined}
                    >
                      {selectedFile ? selectedFile.name : 'Choose PDF'}
                    </p>
                  </div>
                </label>
                <div className="space-y-3">
                  <select value={uploadForm.state} onChange={(e) => setUploadForm(f => ({ ...f, state: e.target.value }))}
                    className="w-full px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white">
                    {metadataOptions?.states.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <select value={uploadForm.document_type} onChange={(e) => setUploadForm(f => ({ ...f, document_type: e.target.value }))}
                    className="w-full px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white">
                    {metadataOptions?.document_types.map((dt) => <option key={dt.value} value={dt.value}>{dt.label}</option>)}
                  </select>
                </div>
                <div className="space-y-3">
                  <select value={uploadForm.category} onChange={(e) => setUploadForm(f => ({ ...f, category: e.target.value }))}
                    className="w-full px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white">
                    {metadataOptions?.categories.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                  </select>
                  <button onClick={handleUpload} disabled={!selectedFile || uploading}
                    className="w-full flex items-center justify-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg text-white font-medium">
                    {uploading ? <><Loader2 className="w-4 h-4 animate-spin" /> Indexing...</> : <><Upload className="w-4 h-4" /> Upload</>}
                  </button>
                </div>
              </div>
            </div>

            {/* Documents List */}
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 overflow-hidden">
              <div className="flex flex-col sm:flex-row sm:items-center gap-4 p-4 border-b border-slate-700">
                <div className="flex-1 min-w-0 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <input type="text" placeholder="Search documents..." value={docsSearch}
                    onChange={(e) => { setDocsSearch(e.target.value); setDocsPage(1); }}
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:outline-none" />
                </div>
                <div className="flex flex-wrap gap-2 sm:gap-3">
                  <select value={docsStateFilter} onChange={(e) => { setDocsStateFilter(e.target.value); setDocsPage(1); }}
                    className="min-w-[140px] flex-1 sm:flex-none px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white">
                    <option value="">All States</option>
                    {metadataOptions?.states.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <select value={docsDocTypeFilter} onChange={(e) => { setDocsDocTypeFilter(e.target.value); setDocsPage(1); }}
                    className="min-w-[160px] flex-1 sm:flex-none px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white">
                    <option value="">All document types</option>
                    {metadataOptions?.document_types.map((dt) => (
                      <option key={dt.value} value={dt.value}>{dt.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {docsLoading && documents.length === 0 ? (
                  // Skeleton loading for document cards
                  <>
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                      <div key={i} className="bg-slate-700/30 rounded-xl p-4 border border-slate-600 animate-pulse">
                        <div className="flex items-center gap-3 mb-3">
                          <div className="w-10 h-10 bg-slate-700 rounded-lg"></div>
                          <div className="flex-1">
                            <div className="h-4 bg-slate-700 rounded w-3/4 mb-1"></div>
                            <div className="h-3 bg-slate-700 rounded w-1/2"></div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="h-4 bg-slate-700 rounded w-full"></div>
                          <div className="h-4 bg-slate-700 rounded w-full"></div>
                          <div className="h-4 bg-slate-700 rounded w-2/3"></div>
                        </div>
                      </div>
                    ))}
                  </>
                ) : (
                documents.map((doc) => (
                  <div key={doc.id} className="bg-slate-700/30 rounded-xl p-4 border border-slate-600 hover:border-blue-500/50 overflow-hidden">
                    <div className="flex items-start justify-between mb-3 gap-2">
                      <div className="flex items-center gap-3 min-w-0 flex-1">
                        <div className="bg-purple-500/10 p-2 rounded-lg flex-shrink-0">
                          <FileText className="w-5 h-5 text-purple-400" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-white font-medium truncate" title={doc.original_filename}>{doc.original_filename}</p>
                          <p className="text-slate-500 text-xs">{doc.file_id}</p>
                        </div>
                      </div>
                      <div className="flex gap-1 flex-shrink-0">
                        <button
                          onClick={() => handleViewDocument(doc)}
                          className="p-1.5 rounded hover:bg-indigo-500/10"
                          title="View document"
                        >
                          <Eye className="w-4 h-4 text-indigo-400" />
                        </button>
                        <button
                          onClick={() => handleDownloadDocument(doc)}
                          className="p-1.5 rounded hover:bg-emerald-500/10"
                          title="Download document"
                        >
                          <Download className="w-4 h-4 text-emerald-400" />
                        </button>
                        <button 
                          onClick={() => handleReindexDocument(doc.id, doc.original_filename)} 
                          disabled={reindexingDocs.has(doc.id)}
                          className="p-1.5 rounded hover:bg-blue-500/10 disabled:opacity-50"
                          title="Reindex with updated categories"
                        >
                          {reindexingDocs.has(doc.id) ? (
                            <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                          ) : (
                            <RefreshCw className="w-4 h-4 text-blue-400" />
                          )}
                        </button>
                        <button onClick={() => handleDeleteDocument(doc.id, doc.original_filename)} className="p-1.5 hover:bg-red-500/10 rounded">
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      </div>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between gap-2 items-start">
                        <span className="text-slate-400 shrink-0">Document type</span>
                        <span className="text-right text-violet-300 font-medium text-xs leading-snug max-w-[65%]" title={doc.document_type}>
                          {metadataOptions?.document_types.find((d) => d.value === doc.document_type)?.label ?? doc.document_type}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2 items-start">
                        <span className="text-slate-400 shrink-0">Sub-category</span>
                        <span className="text-right text-amber-200/90 text-xs font-medium max-w-[65%]" title={doc.category}>
                          {metadataOptions?.categories.find((c) => c.value === doc.category)?.label ?? doc.category}
                        </span>
                      </div>
                      <div className="flex justify-between"><span className="text-slate-400">State</span><span className="text-blue-400">{doc.state}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Pages</span><span className="text-white">{doc.total_pages}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Vectors</span><span className="text-green-400">{doc.total_vectors}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Size</span><span className="text-white">{doc.file_size_kb.toFixed(1)} KB</span></div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-slate-600 flex items-center justify-between">
                      <span className={`text-xs px-2 py-1 rounded ${
                        doc.index_status === 'indexed' ? 'bg-green-500/10 text-green-400' :
                        doc.index_status === 'deleted' ? 'bg-red-500/10 text-red-400' : 'bg-yellow-500/10 text-yellow-400'
                      }`}>{doc.index_status}</span>
                      <span className="text-slate-500 text-xs">{new Date(doc.indexed_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))
                )}
                {!docsLoading && documents.length === 0 && (
                  <div className="col-span-full py-12 text-center">
                    <Layers className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-400">No documents indexed yet</p>
                  </div>
                )}
              </div>

              {documents.length > 0 && (
                <div className="flex items-center justify-between px-6 py-4 border-t border-slate-700">
                  <p className="text-sm text-slate-400">{docsTotal} documents</p>
                  <div className="flex gap-2">
                    <button onClick={() => setDocsPage(p => Math.max(1, p - 1))} disabled={docsPage === 1}
                      className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                      <ChevronLeft className="w-4 h-4 text-white" />
                    </button>
                    <button onClick={() => setDocsPage(p => Math.min(docsTotalPages, p + 1))} disabled={docsPage === docsTotalPages}
                      className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                      <ChevronRight className="w-4 h-4 text-white" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* FAQs Tab */}
        {activeTab === 'faqs' && (
          <div className="space-y-6">
            {/* FAQ Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {faqsLoading && !faqStats ? (
                <>
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="bg-slate-800/50 rounded-xl border border-slate-700 p-4 animate-pulse">
                      <div className="h-4 bg-slate-700 rounded w-20 mb-2"></div>
                      <div className="h-8 bg-slate-700 rounded w-12"></div>
                    </div>
                  ))}
                </>
              ) : (
                <>
                  <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-4">
                    <p className="text-slate-400 text-sm">Total FAQs</p>
                    <p className="text-2xl font-bold text-white">{faqStats?.total || 0}</p>
                  </div>
                  <div className="bg-orange-500/10 rounded-xl border border-orange-500/20 p-4">
                    <p className="text-orange-400 text-sm">Pending Review</p>
                    <p className="text-2xl font-bold text-orange-400">{faqStats?.pending_review || 0}</p>
                  </div>
                  <div className="bg-green-500/10 rounded-xl border border-green-500/20 p-4">
                    <p className="text-green-400 text-sm">Approved</p>
                    <p className="text-2xl font-bold text-green-400">{faqStats?.approved || 0}</p>
                  </div>
                  <div className="bg-red-500/10 rounded-xl border border-red-500/20 p-4">
                    <p className="text-red-400 text-sm">Rejected</p>
                    <p className="text-2xl font-bold text-red-400">{faqStats?.rejected || 0}</p>
                  </div>
                </>
              )}
            </div>

            {/* Auto-Learning Control */}
            <div className={`rounded-2xl border p-4 transition-all duration-300 ${
              autoLearningEnabled 
                ? 'bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border-emerald-500/30' 
                : 'bg-gradient-to-r from-amber-500/10 to-orange-500/10 border-amber-500/30'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl ${
                    autoLearningEnabled 
                      ? 'bg-emerald-500/20' 
                      : 'bg-amber-500/20'
                  }`}>
                    {autoLearningEnabled ? (
                      <Sparkles className="w-6 h-6 text-emerald-400" />
                    ) : (
                      <AlertCircle className="w-6 h-6 text-amber-400" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-white font-semibold flex items-center gap-2">
                      Auto-Learning
                      <span className={`px-2 py-0.5 text-xs rounded-full ${
                        autoLearningEnabled 
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : 'bg-amber-500/20 text-amber-400'
                      }`}>
                        {autoLearningEnabled ? 'Active' : 'Paused'}
                      </span>
                    </h3>
                    <p className="text-slate-400 text-sm mt-0.5">
                      {autoLearningEnabled 
                        ? 'System is capturing Q&A pairs from chat for review' 
                        : 'Auto-capture is paused - no new FAQs will be queued'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {autoLearningUpdatedAt && (
                    <span className="text-xs text-slate-500 hidden md:block">
                      Last updated: {new Date(autoLearningUpdatedAt).toLocaleDateString()}
                    </span>
                  )}
                  <button
                    onClick={toggleAutoLearning}
                    disabled={autoLearningLoading}
                    className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${
                      autoLearningEnabled 
                        ? 'bg-emerald-500 focus:ring-emerald-500' 
                        : 'bg-slate-600 focus:ring-slate-500'
                    }`}
                  >
                    <span className="sr-only">Toggle auto-learning</span>
                    <span
                      className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-lg transition-transform duration-300 ${
                        autoLearningEnabled ? 'translate-x-8' : 'translate-x-1'
                      }`}
                    >
                      {autoLearningLoading && (
                        <Loader2 className="w-5 h-5 text-slate-400 animate-spin" />
                      )}
                    </span>
                  </button>
                </div>
              </div>
            </div>

            {/* Web Fallback Control */}
            <div className={`rounded-2xl border p-4 transition-all duration-300 ${
              webFallbackEnabled 
                ? 'bg-gradient-to-r from-indigo-500/10 to-blue-500/10 border-indigo-500/30' 
                : 'bg-gradient-to-r from-slate-500/10 to-slate-600/10 border-slate-500/30'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl ${
                    webFallbackEnabled 
                      ? 'bg-indigo-500/20' 
                      : 'bg-slate-500/20'
                  }`}>
                    <Globe className={`w-6 h-6 ${webFallbackEnabled ? 'text-indigo-400' : 'text-slate-400'}`} />
                  </div>
                  <div>
                    <h3 className="text-white font-semibold flex items-center gap-2">
                      Web Search Fallback
                      <span className={`px-2 py-0.5 text-xs rounded-full ${
                        webFallbackEnabled 
                          ? 'bg-indigo-500/20 text-indigo-400' 
                          : 'bg-slate-500/20 text-slate-400'
                      }`}>
                        {webFallbackEnabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </h3>
                    <p className="text-slate-400 text-sm mt-0.5">
                      {webFallbackEnabled
                        ? 'If RAG has no result for NEET queries, system will try web search as fallback'
                        : 'System uses RAG-only mode and returns the standard "not in database" message'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {webFallbackUpdatedAt && (
                    <span className="text-xs text-slate-500 hidden md:block">
                      Last updated: {new Date(webFallbackUpdatedAt).toLocaleDateString()}
                    </span>
                  )}
                  <button
                    onClick={toggleWebFallback}
                    disabled={webFallbackLoading}
                    className={`relative inline-flex h-7 w-14 items-center rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${
                      webFallbackEnabled 
                        ? 'bg-indigo-500 focus:ring-indigo-500' 
                        : 'bg-slate-600 focus:ring-slate-500'
                    }`}
                  >
                    <span className="sr-only">Toggle web fallback</span>
                    <span
                      className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-lg transition-transform duration-300 ${
                        webFallbackEnabled ? 'translate-x-8' : 'translate-x-1'
                      }`}
                    >
                      {webFallbackLoading && (
                        <Loader2 className="w-5 h-5 text-slate-400 animate-spin" />
                      )}
                    </span>
                  </button>
                </div>
              </div>
            </div>

            {/* Actions & Filters */}
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-4">
              <div className="flex flex-wrap gap-4 items-center">
                <div className="flex-1 min-w-[200px] relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search FAQs..."
                    value={faqsSearch}
                    onChange={(e) => { setFaqsSearch(e.target.value); setFaqsPage(1); }}
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
                <select value={faqsStatusFilter} onChange={(e) => { setFaqsStatusFilter(e.target.value); setFaqsPage(1); }}
                  className="px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:outline-none">
                  <option value="">All Status</option>
                  <option value="pending">Pending</option>
                  <option value="approved">Approved</option>
                  <option value="rejected">Rejected</option>
                  <option value="modified">Modified</option>
                </select>
                <button
                  onClick={() => setShowCreateFaq(true)}
                  className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
                >
                  <Plus className="w-4 h-4" /> Add FAQ
                </button>
                <button
                  onClick={() => setShowBulkUpload(true)}
                  className="flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg"
                >
                  <FileUp className="w-4 h-4" /> Bulk Upload
                </button>
              </div>
            </div>

            {/* FAQ List */}
            <div className="space-y-4">
              {faqsLoading && faqs.length === 0 ? (
                // Skeleton loading for FAQ cards
                <>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="bg-slate-800/50 rounded-2xl border border-slate-700 p-5 animate-pulse">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="h-5 bg-slate-700 rounded-full w-16"></div>
                        <div className="h-5 bg-slate-700 rounded-full w-20"></div>
                      </div>
                      <div className="h-5 bg-slate-700 rounded w-3/4 mb-2"></div>
                      <div className="h-4 bg-slate-700 rounded w-full mb-1"></div>
                      <div className="h-4 bg-slate-700 rounded w-2/3"></div>
                    </div>
                  ))}
                </>
              ) : (
                faqs.map((faq) => (
                <div key={faq.id} className="bg-slate-800/50 rounded-2xl border border-slate-700 p-5 hover:border-slate-600">
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                          faq.status === 'pending' ? 'bg-orange-500/10 text-orange-400' :
                          faq.status === 'approved' || faq.status === 'modified' ? 'bg-green-500/10 text-green-400' :
                          'bg-red-500/10 text-red-400'
                        }`}>
                          {faq.status}
                        </span>
                        {faq.detected_state && (
                          <span className="px-2 py-0.5 text-xs bg-blue-500/10 text-blue-400 rounded-full">{faq.detected_state}</span>
                        )}
                        {faq.detected_category && (
                          <span className="px-2 py-0.5 text-xs bg-purple-500/10 text-purple-400 rounded-full">{faq.detected_category}</span>
                        )}
                        {faq.occurrence_count > 1 && (
                          <span className="text-xs text-slate-500">Asked {faq.occurrence_count}x</span>
                        )}
                      </div>
                      <p className="text-white font-medium mb-2">{faq.question}</p>
                      <p className="text-slate-400 text-sm line-clamp-2">
                        {faq.modified_answer || faq.original_answer}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      {faq.status === 'pending' && (
                        <>
                          <button
                            onClick={() => handleReviewFaq(faq.id, 'approve')}
                            disabled={reviewingFaqAction?.id === faq.id}
                            className="p-2 hover:bg-green-500/10 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Approve & Vectorize"
                          >
                            {reviewingFaqAction?.id === faq.id && reviewingFaqAction?.action === 'approve' ? (
                              <Loader2 className="w-4 h-4 text-green-400 animate-spin" />
                            ) : (
                              <Check className="w-4 h-4 text-green-400" />
                            )}
                          </button>
                          <button
                            onClick={() => setReviewingFaq(faq)}
                            disabled={reviewingFaqAction?.id === faq.id}
                            className="p-2 hover:bg-blue-500/10 rounded-lg disabled:opacity-50"
                            title="Review & Modify"
                          >
                            <Edit2 className="w-4 h-4 text-blue-400" />
                          </button>
                          <button
                            onClick={() => handleReviewFaq(faq.id, 'reject')}
                            disabled={reviewingFaqAction?.id === faq.id}
                            className="p-2 hover:bg-red-500/10 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Reject"
                          >
                            {reviewingFaqAction?.id === faq.id && reviewingFaqAction?.action === 'reject' ? (
                              <Loader2 className="w-4 h-4 text-red-400 animate-spin" />
                            ) : (
                              <XCircle className="w-4 h-4 text-red-400" />
                            )}
                          </button>
                        </>
                      )}
                      <button
                        onClick={() => setReviewingFaq(faq)}
                        className="p-2 hover:bg-slate-700 rounded-lg"
                        title="View Details"
                      >
                        <Eye className="w-4 h-4 text-slate-400" />
                      </button>
                      <button
                        onClick={() => handleDeleteFaq(faq.id)}
                        className="p-2 hover:bg-red-500/10 rounded-lg"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 pt-3 border-t border-slate-700">
                    <span>Created: {new Date(faq.created_at).toLocaleDateString()}</span>
                    {faq.reviewed_at && <span>Reviewed: {new Date(faq.reviewed_at).toLocaleDateString()}</span>}
                    {faq.faq_vector_id && <span className="text-green-400">✓ Vectorized</span>}
                  </div>
                </div>
              ))
              )}
              
              {!faqsLoading && faqs.length === 0 && (
                <div className="bg-slate-800/50 rounded-2xl border border-slate-700 py-12 text-center">
                  <MessageSquare className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No FAQs found</p>
                  <p className="text-slate-500 text-sm mt-1">Create FAQs manually or wait for auto-learning from chat</p>
                </div>
              )}
            </div>

            {/* Pagination */}
            {faqs.length > 0 && (
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-400">{faqsTotal} FAQs total</p>
                <div className="flex gap-2">
                  <button onClick={() => setFaqsPage(p => Math.max(1, p - 1))} disabled={faqsPage === 1}
                    className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                    <ChevronLeft className="w-4 h-4 text-white" />
                  </button>
                  <span className="px-3 py-2 text-slate-400">Page {faqsPage} of {faqsTotalPages || 1}</span>
                  <button onClick={() => setFaqsPage(p => Math.min(faqsTotalPages || 1, p + 1))} disabled={faqsPage >= (faqsTotalPages || 1)}
                    className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg">
                    <ChevronRight className="w-4 h-4 text-white" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Edit User Modal */}
      {editingUser && (
        <EditUserModal user={editingUser} onClose={() => setEditingUser(null)}
          onSave={(updates) => handleUpdateUser(editingUser.id, updates)} isSuperAdmin={user?.role === 'super_admin'} />
      )}

      {/* Create FAQ Modal */}
      {showCreateFaq && (
        <CreateFAQModal
          onClose={() => setShowCreateFaq(false)}
          onCreate={handleCreateFaq}
        />
      )}

      {/* Review FAQ Modal */}
      {reviewingFaq && (
        <ReviewFAQModal
          faq={reviewingFaq}
          onClose={() => setReviewingFaq(null)}
          onReview={handleReviewFaq}
        />
      )}

      {/* Bulk Upload Modal */}
      {showBulkUpload && (
        <BulkUploadFAQModal
          onClose={() => setShowBulkUpload(false)}
          onUpload={handleBulkUploadFaqs}
        />
      )}
    </div>
  );
}

// Components
function StatCard({ title, value, icon: Icon, color, subtitle }: { title: string; value: string | number; icon: any; color: 'blue' | 'green' | 'purple' | 'orange'; subtitle: string; }) {
  const colors = { blue: 'bg-blue-500/10 text-blue-500 dark:text-blue-400', green: 'bg-green-500/10 text-green-500 dark:text-green-400', purple: 'bg-purple-500/10 text-purple-500 dark:text-purple-400', orange: 'bg-orange-500/10 text-orange-500 dark:text-orange-400' };
  return (
    <div className="bg-white dark:bg-slate-800/50 rounded-2xl border border-gray-200 dark:border-slate-700 p-6 hover:border-gray-300 dark:hover:border-slate-600">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-500 dark:text-slate-400 text-sm">{title}</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white mt-1">{value}</p>
          <p className={`text-sm mt-1 ${colors[color].split(' ')[1]} dark:${colors[color].split(' ')[2]}`}>{subtitle}</p>
        </div>
        <div className={`p-3 rounded-xl ${colors[color].split(' ')[0]}`}>
          <Icon className={`w-6 h-6 ${colors[color].split(' ')[1]} dark:${colors[color].split(' ')[2]}`} />
        </div>
      </div>
    </div>
  );
}

function EditUserModal({ user, onClose, onSave, isSuperAdmin }: { user: User; onClose: () => void; onSave: (updates: Partial<User>) => void; isSuperAdmin: boolean; }) {
  const [form, setForm] = useState({ full_name: user.full_name, email: user.email, phone: user.phone || '', age: user.age?.toString() || '', role: user.role, is_active: user.is_active, is_verified: user.is_verified });

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-200 dark:border-slate-700 w-full max-w-md">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Edit User</h3>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg"><X className="w-4 h-4 text-gray-500 dark:text-slate-400" /></button>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); onSave({ full_name: form.full_name, email: form.email, phone: form.phone || null, age: form.age ? parseInt(form.age) : null, role: form.role, is_active: form.is_active, is_verified: form.is_verified }); }} className="p-6 space-y-4">
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Full Name</label>
            <input type="text" value={form.full_name} onChange={(e) => setForm(f => ({ ...f, full_name: e.target.value }))}
              className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Email</label>
            <input type="email" value={form.email} onChange={(e) => setForm(f => ({ ...f, email: e.target.value }))}
              className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Phone</label>
              <input type="text" value={form.phone} onChange={(e) => setForm(f => ({ ...f, phone: e.target.value }))}
                className="w-full px-4 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white" />
            </div>
            <div>
              <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Age</label>
              <input type="number" value={form.age} onChange={(e) => setForm(f => ({ ...f, age: e.target.value }))}
                className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white" />
            </div>
          </div>
          {isSuperAdmin && (
            <div>
              <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Role</label>
              <select value={form.role} onChange={(e) => setForm(f => ({ ...f, role: e.target.value }))}
                className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white">
                <option value="student">Student</option>
                <option value="admin">Admin</option>
                <option value="super_admin">Super Admin</option>
              </select>
            </div>
          )}
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={form.is_active} onChange={(e) => setForm(f => ({ ...f, is_active: e.target.checked }))}
                className="w-4 h-4 rounded border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-blue-500" />
              <span className="text-gray-600 dark:text-slate-300">Active</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={form.is_verified} onChange={(e) => setForm(f => ({ ...f, is_verified: e.target.checked }))}
                className="w-4 h-4 rounded border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-blue-500" />
              <span className="text-gray-600 dark:text-slate-300">Verified</span>
            </label>
          </div>
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2.5 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg text-gray-900 dark:text-white">Cancel</button>
            <button type="submit" className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium">Save</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Create FAQ Modal
function CreateFAQModal({ onClose, onCreate }: { onClose: () => void; onCreate: (q: string, a: string, state?: string, category?: string) => void }) {
  const [form, setForm] = useState({ question: '', answer: '', state: '', category: '' });

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-200 dark:border-slate-700 w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-slate-700 sticky top-0 bg-white dark:bg-slate-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Create FAQ</h3>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg"><X className="w-4 h-4 text-gray-500 dark:text-slate-400" /></button>
        </div>
        <form onSubmit={(e) => { e.preventDefault(); onCreate(form.question, form.answer, form.state || undefined, form.category || undefined); }} className="p-6 space-y-4">
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Question *</label>
            <input
              type="text"
              value={form.question}
              onChange={(e) => setForm(f => ({ ...f, question: e.target.value }))}
              className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white"
              placeholder="What is the age limit for NEET?"
              required
            />
          </div>
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Answer *</label>
            <textarea
              value={form.answer}
              onChange={(e) => setForm(f => ({ ...f, answer: e.target.value }))}
              className="w-full px-4 py-3 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white min-h-[120px]"
              placeholder="The age limit for NEET UG 2026 is..."
              required
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">State (optional)</label>
              <input
                type="text"
                value={form.state}
                onChange={(e) => setForm(f => ({ ...f, state: e.target.value }))}
                className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white"
                placeholder="All-India"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Category (optional)</label>
              <input
                type="text"
                value={form.category}
                onChange={(e) => setForm(f => ({ ...f, category: e.target.value }))}
                className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white"
                placeholder="eligibility"
              />
            </div>
          </div>
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2.5 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg text-gray-900 dark:text-white">Cancel</button>
            <button type="submit" className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium">Create FAQ</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Review FAQ Modal
function ReviewFAQModal({ faq, onClose, onReview }: { faq: FAQ; onClose: () => void; onReview: (id: number, action: 'approve' | 'reject' | 'modify', modified?: string, notes?: string) => void }) {
  const [modifiedAnswer, setModifiedAnswer] = useState(faq.modified_answer || faq.original_answer);
  const [reviewNotes, setReviewNotes] = useState('');

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-200 dark:border-slate-700 w-full max-w-3xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-slate-700 sticky top-0 bg-white dark:bg-slate-800">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Review FAQ</h3>
            <div className="flex gap-2 mt-1">
              <span className={`px-2 py-0.5 text-xs rounded-full ${
                faq.status === 'pending' ? 'bg-orange-500/10 text-orange-500 dark:text-orange-400' :
                faq.status === 'approved' || faq.status === 'modified' ? 'bg-green-500/10 text-green-500 dark:text-green-400' :
                'bg-red-500/10 text-red-500 dark:text-red-400'
              }`}>{faq.status}</span>
              {faq.occurrence_count > 1 && <span className="text-xs text-gray-400 dark:text-slate-500">Asked {faq.occurrence_count}x</span>}
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg"><X className="w-4 h-4 text-gray-500 dark:text-slate-400" /></button>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Question</label>
            <p className="px-4 py-3 bg-gray-50 dark:bg-slate-700/30 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white">{faq.question}</p>
          </div>
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Original Answer</label>
            <p className="px-4 py-3 bg-gray-50 dark:bg-slate-700/30 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-600 dark:text-slate-300 text-sm whitespace-pre-wrap">{faq.original_answer}</p>
          </div>
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Modified Answer (edit if needed)</label>
            <textarea
              value={modifiedAnswer}
              onChange={(e) => setModifiedAnswer(e.target.value)}
              className="w-full px-4 py-3 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white min-h-[150px]"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">Review Notes (optional)</label>
            <input
              type="text"
              value={reviewNotes}
              onChange={(e) => setReviewNotes(e.target.value)}
              className="w-full px-4 py-2.5 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white"
              placeholder="Any notes about this review..."
            />
          </div>
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="px-4 py-2.5 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg text-gray-900 dark:text-white">Cancel</button>
            <button
              onClick={() => onReview(faq.id, 'reject', undefined, reviewNotes)}
              className="px-4 py-2.5 bg-red-600 hover:bg-red-700 rounded-lg text-white"
            >
              Reject
            </button>
            <button
              onClick={() => onReview(faq.id, 'modify', modifiedAnswer, reviewNotes)}
              className="px-4 py-2.5 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-white"
            >
              Save Modified
            </button>
            <button
              onClick={() => onReview(faq.id, 'approve', undefined, reviewNotes)}
              className="px-4 py-2.5 bg-green-600 hover:bg-green-700 rounded-lg text-white"
            >
              Approve
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Bulk Upload FAQ Modal
function BulkUploadFAQModal({ onClose, onUpload }: { onClose: () => void; onUpload: (faqs: { question: string; answer: string; state?: string; category?: string }[], autoApprove: boolean) => void }) {
  const [jsonText, setJsonText] = useState('');
  const [autoApprove, setAutoApprove] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = () => {
    try {
      const parsed = JSON.parse(jsonText);
      if (!Array.isArray(parsed)) {
        setError('JSON must be an array');
        return;
      }
      for (const item of parsed) {
        if (!item.question || !item.answer) {
          setError('Each item must have "question" and "answer" fields');
          return;
        }
      }
      onUpload(parsed, autoApprove);
    } catch (e) {
      setError('Invalid JSON format');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-200 dark:border-slate-700 w-full max-w-3xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-slate-700 sticky top-0 bg-white dark:bg-slate-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Bulk Upload FAQs</h3>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg"><X className="w-4 h-4 text-gray-500 dark:text-slate-400" /></button>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm text-gray-500 dark:text-slate-400 mb-1">JSON Array of FAQs</label>
            <textarea
              value={jsonText}
              onChange={(e) => { setJsonText(e.target.value); setError(null); }}
              className="w-full px-4 py-3 bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600 rounded-lg text-gray-900 dark:text-white font-mono text-sm min-h-[250px]"
              placeholder={`[
  {
    "question": "What is the age limit for NEET?",
    "answer": "17-25 years as on 31st December 2026",
    "state": "All-India",
    "category": "eligibility"
  },
  ...
]`}
            />
            {error && <p className="text-red-500 dark:text-red-400 text-sm mt-1">{error}</p>}
          </div>
          <div className="text-sm text-gray-400 dark:text-slate-500">
            <p>Format: Array of objects with required "question" and "answer" fields.</p>
            <p>Optional: "state" and "category" fields for organization.</p>
          </div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={autoApprove} onChange={(e) => setAutoApprove(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-blue-500" />
            <span className="text-gray-600 dark:text-slate-300">Auto-approve all FAQs</span>
          </label>
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2.5 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg text-gray-900 dark:text-white">Cancel</button>
            <button onClick={handleUpload} className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium">Upload</button>
          </div>
        </div>
      </div>
    </div>
  );
}
