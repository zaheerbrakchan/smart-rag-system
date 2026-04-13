'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import ThemeToggle from '@/components/ThemeToggle';
import {
  requestForgotPassword,
  verifyForgotPasswordOtp,
  resetPasswordWithToken,
} from '@/services/authApi';
import {
  GraduationCap,
  User,
  Lock,
  Eye,
  EyeOff,
  Phone,
  ArrowRight,
  Loader2,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

type Step = 'identifier' | 'otp' | 'password';

export default function ForgotPasswordPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  const [step, setStep] = useState<Step>('identifier');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const [identifier, setIdentifier] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [devOtpHint, setDevOtpHint] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [infoMessage, setInfoMessage] = useState('');

  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.push('/');
    }
  }, [authLoading, isAuthenticated, router]);

  const handleRequestOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setInfoMessage('');
    setIsLoading(true);
    try {
      const res = await requestForgotPassword(identifier);
      setInfoMessage(res.message);
      if (res.otp) setDevOtpHint(res.otp);
      setStep('otp');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      const res = await verifyForgotPasswordOtp(phone, otp);
      setResetToken(res.reset_token);
      setStep('password');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setIsLoading(true);
    try {
      await resetPasswordWithToken(resetToken, newPassword);
      router.push('/login?reset=success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Reset failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 flex flex-col">
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-blue-100 dark:border-slate-700 px-6 py-3 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
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
              href="/login"
              className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
            >
              Back to sign in
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center px-4 py-8">
        <div className="w-full max-w-md">
          <div className="flex items-center justify-center gap-2 mb-6 text-sm text-gray-500 dark:text-gray-400">
            <span className={step === 'identifier' ? 'font-semibold text-blue-600 dark:text-blue-400' : ''}>
              1. Account
            </span>
            <span>→</span>
            <span className={step === 'otp' ? 'font-semibold text-blue-600 dark:text-blue-400' : ''}>2. OTP</span>
            <span>→</span>
            <span className={step === 'password' ? 'font-semibold text-blue-600 dark:text-blue-400' : ''}>
              3. New password
            </span>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl dark:shadow-2xl dark:shadow-black/20 p-6">
            <div className="text-center mb-5">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl mb-3">
                <Lock className="w-7 h-7 text-white" />
              </div>
              <h2 className="text-xl font-bold text-gray-800 dark:text-white">Reset password</h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
                {step === 'identifier' && 'Enter your username or email — we’ll text an OTP to your registered mobile.'}
                {step === 'otp' && 'Enter the mobile number and the OTP you received.'}
                {step === 'password' && 'Choose a new password for your account.'}
              </p>
            </div>

            {step === 'identifier' && (
              <form onSubmit={handleRequestOtp} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    Username or email
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="text"
                      value={identifier}
                      onChange={(e) => setIdentifier(e.target.value)}
                      placeholder="Same as sign-in"
                      className="w-full pl-11 pr-4 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white"
                      required
                    />
                  </div>
                </div>
                {error && (
                  <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm bg-red-50 dark:bg-red-500/10 p-3 rounded-lg">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    {error}
                  </div>
                )}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Send OTP <ArrowRight className="w-5 h-5" /></>}
                </button>
              </form>
            )}

            {step === 'otp' && (
              <form onSubmit={handleVerifyOtp} className="space-y-4">
                {infoMessage && (
                  <div className="flex items-start gap-2 text-sm text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-500/10 p-3 rounded-lg">
                    <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>{infoMessage}</span>
                  </div>
                )}
                {devOtpHint && (
                  <p className="text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-500/10 p-2 rounded-lg">
                    Dev mode OTP: <strong>{devOtpHint}</strong>
                  </p>
                )}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    Registered mobile number
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      placeholder="+91 9876543210"
                      className="w-full pl-11 pr-4 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white"
                      required
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    OTP
                  </label>
                  <input
                    type="text"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 8))}
                    placeholder="Enter OTP"
                    className="w-full px-4 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white tracking-widest"
                    required
                  />
                </div>
                {error && (
                  <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm bg-red-50 dark:bg-red-500/10 p-3 rounded-lg">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    {error}
                  </div>
                )}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Verify & continue <ArrowRight className="w-5 h-5" /></>}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setStep('identifier');
                    setError('');
                    setOtp('');
                  }}
                  className="w-full text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700"
                >
                  ← Change email / username
                </button>
              </form>
            )}

            {step === 'password' && (
              <form onSubmit={handleResetPassword} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    New password
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      className="w-full pl-11 pr-12 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white"
                      required
                      minLength={8}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    Confirm password
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="w-full pl-11 pr-4 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white"
                      required
                      minLength={8}
                    />
                  </div>
                </div>
                {error && (
                  <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm bg-red-50 dark:bg-red-500/10 p-3 rounded-lg">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    {error}
                  </div>
                )}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Update password</>}
                </button>
              </form>
            )}
          </div>

          <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-4">
            Remember your password?{' '}
            <Link href="/login" className="text-blue-600 dark:text-blue-400 font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}
