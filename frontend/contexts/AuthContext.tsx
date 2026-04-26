'use client';

import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import { User, AuthResponse } from '@/types';
import * as authApi from '@/services/authApi';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (phone: string, verificationToken: string) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: (reason?: 'manual' | 'inactive') => void;
  sendOtp: (phone: string, purpose?: 'registration' | 'login' | 'password_reset') => Promise<{ success: boolean; message: string; otp?: string }>;
  verifyOtp: (phone: string, otp: string, purpose?: 'registration' | 'login' | 'password_reset') => Promise<{ success: boolean; message: string; verification_token?: string }>;
}

interface RegisterData {
  full_name: string;
  phone: string;
  verification_token: string;
  email?: string;
  state_or_ut?: string;
  city?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = 'auth_tokens';
const USER_KEY = 'auth_user';
const LOGOUT_NOTICE_KEY = 'auth_logout_notice';
const INACTIVITY_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const inactivityTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load user from localStorage on mount
  useEffect(() => {
    const loadUser = async () => {
      // Set a timeout to prevent infinite loading
      const timeout = setTimeout(() => {
        console.log('Auth loading timeout - setting isLoading to false');
        setIsLoading(false);
      }, 5000); // 5 second timeout

      try {
        const storedUser = localStorage.getItem(USER_KEY);
        const storedTokens = localStorage.getItem(TOKEN_KEY);

        if (storedUser && storedTokens) {
          const tokens = JSON.parse(storedTokens);
          
          // Verify token is still valid by fetching current user
          try {
            const currentUser = await authApi.getCurrentUser(tokens.access_token);
            setUser(currentUser);
            setToken(tokens.access_token);
          } catch (error) {
            // Token expired, try refresh
            try {
              const newTokens = await authApi.refreshToken(tokens.refresh_token);
              localStorage.setItem(TOKEN_KEY, JSON.stringify({
                access_token: newTokens.access_token,
                refresh_token: tokens.refresh_token
              }));
              const currentUser = await authApi.getCurrentUser(newTokens.access_token);
              setUser(currentUser);
              setToken(newTokens.access_token);
            } catch (refreshError) {
              // Refresh failed, clear storage
              localStorage.removeItem(TOKEN_KEY);
              localStorage.removeItem(USER_KEY);
              setToken(null);
            }
          }
        }
      } catch (error) {
        console.error('Error loading user:', error);
      } finally {
        clearTimeout(timeout);
        setIsLoading(false);
      }
    };

    loadUser();
  }, []);

  const login = async (phone: string, verificationToken: string) => {
    const response = await authApi.login(phone, verificationToken);
    
    // Store tokens and user
    localStorage.setItem(TOKEN_KEY, JSON.stringify({
      access_token: response.access_token,
      refresh_token: response.refresh_token
    }));
    localStorage.setItem(USER_KEY, JSON.stringify(response.user));
    
    setUser(response.user);
    setToken(response.access_token);
  };

  const register = async (data: RegisterData) => {
    const response = await authApi.register(data);
    
    // Store tokens and user
    localStorage.setItem(TOKEN_KEY, JSON.stringify({
      access_token: response.access_token,
      refresh_token: response.refresh_token
    }));
    localStorage.setItem(USER_KEY, JSON.stringify(response.user));
    
    setUser(response.user);
    setToken(response.access_token);
  };

  const logout = (reason: 'manual' | 'inactive' = 'manual') => {
    if (inactivityTimerRef.current) {
      clearTimeout(inactivityTimerRef.current);
      inactivityTimerRef.current = null;
    }
    // Call logout API (optional, since JWT is stateless)
    const tokens = localStorage.getItem(TOKEN_KEY);
    if (tokens) {
      const { access_token } = JSON.parse(tokens);
      authApi.logout(access_token).catch(() => {});
    }
    
    // Clear storage
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    if (reason === 'inactive') {
      localStorage.setItem(LOGOUT_NOTICE_KEY, 'You were logged out due to 10 minutes of inactivity.');
    } else {
      localStorage.removeItem(LOGOUT_NOTICE_KEY);
    }
    setUser(null);
    setToken(null);
  };

  useEffect(() => {
    if (isLoading || !user || !token) {
      if (inactivityTimerRef.current) {
        clearTimeout(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }
      return;
    }

    const resetInactivityTimer = () => {
      if (inactivityTimerRef.current) {
        clearTimeout(inactivityTimerRef.current);
      }
      inactivityTimerRef.current = setTimeout(() => {
        logout('inactive');
      }, INACTIVITY_TIMEOUT_MS);
    };

    // Start timer immediately after auth is active.
    resetInactivityTimer();

    const events: Array<keyof WindowEventMap> = [
      'mousedown',
      'keydown',
      'scroll',
      'touchstart',
      'click',
    ];
    for (const eventName of events) {
      window.addEventListener(eventName, resetInactivityTimer, { passive: true });
    }

    return () => {
      for (const eventName of events) {
        window.removeEventListener(eventName, resetInactivityTimer);
      }
      if (inactivityTimerRef.current) {
        clearTimeout(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }
    };
  }, [isLoading, user, token]);

  const sendOtp = async (phone: string, purpose: 'registration' | 'login' | 'password_reset' = 'registration') => {
    return await authApi.sendOtp(phone, purpose);
  };

  const verifyOtp = async (phone: string, otp: string, purpose: 'registration' | 'login' | 'password_reset' = 'registration') => {
    return await authApi.verifyOtp(phone, otp, purpose);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        sendOtp,
        verifyOtp,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Helper to get access token
export function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null;
  
  const tokens = localStorage.getItem(TOKEN_KEY);
  if (!tokens) return null;
  
  try {
    const { access_token } = JSON.parse(tokens);
    return access_token;
  } catch {
    return null;
  }
}
