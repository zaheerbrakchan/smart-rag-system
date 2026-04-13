'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import ThemeToggle from '@/components/ThemeToggle';
import { 
  GraduationCap, User, Mail, Phone, Calendar, Target, 
  ArrowLeft, Edit2, Save, X, Loader2, CheckCircle
} from 'lucide-react';

export default function ProfilePage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  
  // Redirect if not authenticated
  if (!isLoading && !isAuthenticated) {
    router.push('/login');
    return null;
  }
  
  if (isLoading) {
    return (
      <div className="h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-blue-100 dark:border-slate-700 px-6 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl shadow-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Get My University
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">NEET UG 2026 Assistant</p>
            </div>
          </Link>
          
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Link 
              href="/"
              className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Chat
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Profile Header */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl dark:shadow-2xl dark:shadow-black/20 overflow-hidden mb-6">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 h-32" />
          <div className="px-8 pb-8">
            <div className="flex items-end -mt-16 mb-4">
              <div className="w-32 h-32 bg-white dark:bg-slate-700 rounded-2xl shadow-lg flex items-center justify-center text-4xl font-bold text-blue-600 dark:text-blue-400 border-4 border-white dark:border-slate-800">
                {user?.full_name?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="ml-6 pb-2">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white">{user?.full_name}</h2>
                <p className="text-gray-500 dark:text-gray-400">@{user?.username}</p>
                <span className={`inline-block mt-2 px-3 py-1 text-sm font-medium rounded-full capitalize ${
                  user?.role === 'admin' || user?.role === 'super_admin' 
                    ? 'bg-purple-100 dark:bg-purple-500/20 text-purple-700 dark:text-purple-400' 
                    : 'bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-400'
                }`}>
                  {user?.role}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Profile Details */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Personal Information */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg dark:shadow-xl dark:shadow-black/10 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Personal Information</h3>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
              >
                {isEditing ? (
                  <>
                    <X className="w-4 h-4" />
                    Cancel
                  </>
                ) : (
                  <>
                    <Edit2 className="w-4 h-4" />
                    Edit
                  </>
                )}
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-blue-50 dark:bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <User className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Full Name</p>
                  <p className="font-medium text-gray-800 dark:text-white">{user?.full_name}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-green-50 dark:bg-green-500/20 rounded-lg flex items-center justify-center">
                  <Mail className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Email</p>
                  <p className="font-medium text-gray-800 dark:text-white">{user?.email}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-purple-50 dark:bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <Phone className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Phone</p>
                  <p className="font-medium text-gray-800 dark:text-white">{user?.phone || 'Not provided'}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-orange-50 dark:bg-orange-500/20 rounded-lg flex items-center justify-center">
                  <Calendar className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Preferred State</p>
                  <p className="font-medium text-gray-800 dark:text-white">{user?.preferences?.preferred_state || 'Not provided'}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Target Exams */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg dark:shadow-xl dark:shadow-black/10 p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-6">Target Exams</h3>
            
            <div className="flex items-start gap-4 mb-6">
              <div className="w-10 h-10 bg-indigo-50 dark:bg-indigo-500/20 rounded-lg flex items-center justify-center">
                <Target className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Preparing For</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-4 py-2 bg-indigo-50 dark:bg-indigo-500/20 text-indigo-700 dark:text-indigo-400 rounded-full text-sm font-medium">
                    NEET UG
                  </span>
                </div>
              </div>
            </div>

            {/* Account Status */}
            <div className="pt-4 border-t border-gray-100 dark:border-slate-700">
              <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Account Status</h4>
              <div className="flex items-center gap-4">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                  user?.is_verified 
                    ? 'bg-green-50 dark:bg-green-500/20 text-green-700 dark:text-green-400' 
                    : 'bg-yellow-50 dark:bg-yellow-500/20 text-yellow-700 dark:text-yellow-400'
                }`}>
                  <CheckCircle className="w-4 h-4" />
                  {user?.is_verified ? 'Verified' : 'Unverified'}
                </div>
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                  user?.is_active 
                    ? 'bg-green-50 dark:bg-green-500/20 text-green-700 dark:text-green-400' 
                    : 'bg-red-50 dark:bg-red-500/20 text-red-700 dark:text-red-400'
                }`}>
                  <CheckCircle className="w-4 h-4" />
                  {user?.is_active ? 'Active' : 'Inactive'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Account Info */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg dark:shadow-xl dark:shadow-black/10 p-6 mt-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Account Information</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Member since: {user?.created_at ? new Date(user.created_at).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            }) : 'N/A'}
          </p>
        </div>
      </main>
    </div>
  );
}
