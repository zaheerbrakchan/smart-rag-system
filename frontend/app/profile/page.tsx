'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { getCutoffProfileOptions } from '@/services/api';
import { updateMyProfile } from '@/services/authApi';
import ThemeToggle from '@/components/ThemeToggle';
import { 
  GraduationCap, User, Mail, Phone, Calendar, MapPin, ListFilter, Tag, 
  ArrowLeft, Edit2, Save, X, Loader2, CheckCircle
} from 'lucide-react';

export default function ProfilePage() {
  const router = useRouter();
  const { user, token, refreshUser, isAuthenticated, isLoading } = useAuth();
  useTheme();
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [states, setStates] = useState<string[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [subCategories, setSubCategories] = useState<string[]>([]);
  const [homeState, setHomeState] = useState('');
  const [category, setCategory] = useState('');
  const [subCategory, setSubCategory] = useState('');

  const profileData = user?.profile_data || {};
  const cutoffProfile = profileData.cutoff_profile || {};
  const email = profileData.email || 'Not provided';

  useEffect(() => {
    setHomeState(cutoffProfile.home_state || '');
    setCategory(cutoffProfile.category || '');
    setSubCategory(cutoffProfile.sub_category || '');
  }, [cutoffProfile.home_state, cutoffProfile.category, cutoffProfile.sub_category]);

  useEffect(() => {
    const loadOptions = async () => {
      try {
        const data = await getCutoffProfileOptions(homeState || undefined, category || undefined);
        setStates(data.states || []);
        setCategories(data.categories || []);
        setSubCategories(data.sub_categories || []);
      } catch (e) {
        // Non-blocking: existing values can still be edited/saved.
      }
    };
    loadOptions();
  }, [homeState, category]);

  const hasChanges = useMemo(() => {
    return (
      homeState !== (cutoffProfile.home_state || '') ||
      category !== (cutoffProfile.category || '') ||
      subCategory !== (cutoffProfile.sub_category || '')
    );
  }, [homeState, category, subCategory, cutoffProfile.home_state, cutoffProfile.category, cutoffProfile.sub_category]);
  
  // Redirect if not authenticated
  if (!isLoading && !isAuthenticated) {
    router.push('/login');
    return null;
  }
  
  if (isLoading) {
    return (
      <div className="h-screen bg-gradient-to-br from-red-50 via-white to-rose-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-red-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-rose-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-red-100 dark:border-slate-700 px-6 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-red-600 to-rose-600 p-2 rounded-xl shadow-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-red-600 to-rose-600 bg-clip-text text-transparent">
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
          <div className="bg-gradient-to-r from-red-600 to-rose-600 h-32" />
          <div className="px-8 pb-8">
            <div className="flex items-end -mt-16 mb-4">
              <div className="w-32 h-32 bg-white dark:bg-slate-700 rounded-2xl shadow-lg flex items-center justify-center text-4xl font-bold text-red-600 dark:text-red-400 border-4 border-white dark:border-slate-800">
                {user?.full_name?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="ml-6 pb-2">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white">{user?.full_name}</h2>
                <p className="text-gray-500 dark:text-gray-400">{user?.phone || 'Student Account'}</p>
                <span className={`inline-block mt-2 px-3 py-1 text-sm font-medium rounded-full capitalize ${
                  user?.role === 'admin' || user?.role === 'super_admin' 
                    ? 'bg-purple-100 dark:bg-purple-500/20 text-purple-700 dark:text-purple-400' 
                    : 'bg-red-100 dark:bg-red-500/20 text-red-700 dark:text-red-400'
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
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-red-50 dark:bg-red-500/20 rounded-lg flex items-center justify-center">
                  <User className="w-5 h-5 text-red-600 dark:text-red-400" />
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
                  <p className="font-medium text-gray-800 dark:text-white">{email}</p>
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
              
            </div>
          </div>

          {/* Counselling Profile */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg dark:shadow-xl dark:shadow-black/10 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Counselling Profile</h3>
              <button
                onClick={() => {
                  setIsEditing(!isEditing);
                  setError(null);
                  setSuccess(null);
                  if (isEditing) {
                    setHomeState(cutoffProfile.home_state || '');
                    setCategory(cutoffProfile.category || '');
                    setSubCategory(cutoffProfile.sub_category || '');
                  }
                }}
                className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
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

            {error && (
              <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300">
                {error}
              </div>
            )}
            {success && (
              <div className="mb-4 rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700 dark:border-green-500/30 dark:bg-green-500/10 dark:text-green-300">
                {success}
              </div>
            )}

            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-rose-50 dark:bg-rose-500/20 rounded-lg flex items-center justify-center">
                  <MapPin className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Home State</p>
                  {isEditing ? (
                    <select
                      value={homeState}
                      onChange={(e) => setHomeState(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 focus:border-red-500 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                    >
                      <option value="">Select home state</option>
                      {states.map((state) => (
                        <option key={state} value={state}>
                          {state}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="font-medium text-gray-800 dark:text-white">{cutoffProfile.home_state || 'Not provided'}</p>
                  )}
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-red-50 dark:bg-red-500/20 rounded-lg flex items-center justify-center">
                  <ListFilter className="w-5 h-5 text-red-600 dark:text-red-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Category</p>
                  {isEditing ? (
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 focus:border-red-500 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                    >
                      <option value="">Select category</option>
                      {categories.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="font-medium text-gray-800 dark:text-white">{cutoffProfile.category || 'Not provided'}</p>
                  )}
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-purple-50 dark:bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <Tag className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Sub-category</p>
                  {isEditing ? (
                    <select
                      value={subCategory}
                      onChange={(e) => setSubCategory(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 focus:border-red-500 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                    >
                      <option value="">Select sub-category</option>
                      {subCategories.map((sub) => (
                        <option key={sub} value={sub}>
                          {sub}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="font-medium text-gray-800 dark:text-white">{cutoffProfile.sub_category || 'Not provided'}</p>
                  )}
                </div>
              </div>
            </div>

            {isEditing && (
              <div className="pt-4 mt-4 border-t border-gray-100 dark:border-slate-700">
                <button
                  onClick={async () => {
                    if (!token) return;
                    setSaving(true);
                    setError(null);
                    setSuccess(null);
                    try {
                      await updateMyProfile(token, {
                        home_state: homeState || '',
                        category: category || '',
                        sub_category: subCategory || '',
                      });
                      await refreshUser();
                      setSuccess('Profile updated successfully.');
                      setIsEditing(false);
                    } catch (e: any) {
                      setError(e?.message || 'Failed to update profile.');
                    } finally {
                      setSaving(false);
                    }
                  }}
                  disabled={saving || !hasChanges}
                  className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                  Save Changes
                </button>
              </div>
            )}

            {/* Account Status */}
            <div className="pt-4 mt-4 border-t border-gray-100 dark:border-slate-700">
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
