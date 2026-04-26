'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import ThemeToggle from '@/components/ThemeToggle';
import { 
  GraduationCap, Phone,
  ArrowRight, Loader2, AlertCircle 
} from 'lucide-react';

type Step = 'phone' | 'otp';
const LOGOUT_NOTICE_KEY = 'auth_logout_notice';

export default function LoginPage() {
  const router = useRouter();
  const { login, sendOtp, verifyOtp, isAuthenticated, isLoading: authLoading } = useAuth();
  
  const [step, setStep] = useState<Step>('phone');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [phone, setPhone] = useState('+91 ');
  const [otp, setOtp] = useState('');
  const [devOtp, setDevOtp] = useState('');
  const [countdown, setCountdown] = useState(0);

  // Redirect if already authenticated
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.push('/');
    }
  }, [authLoading, isAuthenticated, router]);

  useEffect(() => {
    const msg = localStorage.getItem(LOGOUT_NOTICE_KEY);
    if (msg) {
      setNotice(msg);
      localStorage.removeItem(LOGOUT_NOTICE_KEY);
    }
  }, []);

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      const result = await sendOtp(phone, 'login');
      if (result.success) {
        setStep('otp');
        setCountdown(60);
        if (result.otp) setDevOtp(result.otp);
        const interval = setInterval(() => {
          setCountdown((prev) => {
            if (prev <= 1) {
              clearInterval(interval);
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyAndLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const verified = await verifyOtp(phone, otp, 'login');
      if (!verified.success || !verified.verification_token) {
        throw new Error('OTP verification failed');
      }
      localStorage.removeItem(LOGOUT_NOTICE_KEY);
      await login(phone, verified.verification_token);
      router.push('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border-b border-blue-100 dark:border-slate-700 px-6 py-3 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-xl shadow-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Med Buddy
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Powered by Get My University</p>
            </div>
          </Link>
          
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Link 
              href="/register"
              className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
            >
              Don't have an account? Sign up
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 py-4 overflow-auto">
        <div className="w-full max-w-md">
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl dark:shadow-2xl dark:shadow-black/20 p-6">
            <div className="text-center mb-5">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl mb-3">
                <GraduationCap className="w-7 h-7 text-white" />
              </div>
              <p className="text-xs uppercase tracking-wide text-blue-600 dark:text-blue-400 font-semibold mb-1">
                Med Buddy
              </p>
              <h2 className="text-xl font-bold text-gray-800 dark:text-white">Sign In with OTP</h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Use your registered mobile number</p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">Powered by Get My University</p>
            </div>
            
            {step === 'phone' && (
              <form onSubmit={handleSendOtp} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    Mobile Number
                  </label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      placeholder="Enter registered mobile number"
                      className="w-full pl-11 pr-4 py-2.5 border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                {notice && (
                  <div className="flex items-center gap-2 text-amber-700 dark:text-amber-300 text-sm bg-amber-50 dark:bg-amber-500/10 p-3 rounded-lg border border-amber-200 dark:border-amber-500/30">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    {notice}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Send OTP <ArrowRight className="w-5 h-5" /></>}
                </button>
              </form>
            )}

            {step === 'otp' && (
              <form onSubmit={handleVerifyAndLogin} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                    Enter OTP sent to {phone}
                  </label>
                  <input
                    type="text"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    placeholder="000000"
                    className="w-full px-4 py-2.5 text-center text-2xl tracking-widest border border-gray-200 dark:border-slate-600 rounded-xl bg-white dark:bg-slate-700 text-gray-900 dark:text-white placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    maxLength={6}
                    required
                  />
                </div>

                {devOtp && (
                  <div className="text-sm text-yellow-700 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-500/10 p-3 rounded-lg border border-yellow-200 dark:border-yellow-500/30">
                    <strong>Dev OTP:</strong> {devOtp}
                  </div>
                )}
              
              {error && (
                <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm bg-red-50 dark:bg-red-500/10 p-3 rounded-lg">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}
              {notice && (
                <div className="flex items-center gap-2 text-amber-700 dark:text-amber-300 text-sm bg-amber-50 dark:bg-amber-500/10 p-3 rounded-lg border border-amber-200 dark:border-amber-500/30">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {notice}
                </div>
              )}

                <button
                  type="submit"
                  disabled={isLoading || otp.length !== 6}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <>Verify OTP & Sign In <ArrowRight className="w-5 h-5" /></>}
                </button>

                <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                  {countdown > 0 ? (
                    <span>Resend OTP in {countdown}s</span>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendOtp}
                      className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium"
                    >
                      Resend OTP
                    </button>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => { setStep('phone'); setOtp(''); setError(''); }}
                  className="w-full text-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                >
                  Change phone number
                </button>
              </form>
            )}
          </div>
          
          <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-4">
            New to Med Buddy?{' '}
            <Link href="/register" className="text-blue-600 dark:text-blue-400 hover:underline font-medium">
              Create an account
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}
