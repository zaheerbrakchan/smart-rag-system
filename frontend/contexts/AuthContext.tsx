'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, AuthResponse } from '@/types';
import * as authApi from '@/services/authApi';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  sendOtp: (phone: string) => Promise<{ success: boolean; message: string; otp?: string }>;
  verifyOtp: (phone: string, otp: string) => Promise<{ success: boolean; message: string; verification_token?: string }>;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name: string;
  phone: string;
  age?: number;
  target_exams?: string[];
  verification_token: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = 'auth_tokens';
const USER_KEY = 'auth_user';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

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
            } catch (refreshError) {
              // Refresh failed, clear storage
              localStorage.removeItem(TOKEN_KEY);
              localStorage.removeItem(USER_KEY);
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

  const login = async (username: string, password: string) => {
    const response = await authApi.login(username, password);
    
    // Store tokens and user
    localStorage.setItem(TOKEN_KEY, JSON.stringify({
      access_token: response.access_token,
      refresh_token: response.refresh_token
    }));
    localStorage.setItem(USER_KEY, JSON.stringify(response.user));
    
    setUser(response.user);
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
  };

  const logout = () => {
    // Call logout API (optional, since JWT is stateless)
    const tokens = localStorage.getItem(TOKEN_KEY);
    if (tokens) {
      const { access_token } = JSON.parse(tokens);
      authApi.logout(access_token).catch(() => {});
    }
    
    // Clear storage
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setUser(null);
  };

  const sendOtp = async (phone: string) => {
    return await authApi.sendOtp(phone);
  };

  const verifyOtp = async (phone: string, otp: string) => {
    return await authApi.verifyOtp(phone, otp);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
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
