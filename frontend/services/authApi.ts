import { User, AuthResponse } from '@/types';

// Use relative URL to leverage Next.js rewrites (proxies to backend)
// This avoids CORS issues in development
const API_BASE_URL = '/api';

/**
 * Send OTP to phone number
 */
export async function sendOtp(
  phone: string,
  purpose: string = 'registration'
): Promise<{ success: boolean; message: string; otp?: string }> {
  try {
    console.log('Sending OTP to:', phone);
    const response = await fetch(`${API_BASE_URL}/auth/send-otp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ phone, purpose }),
    });

    console.log('OTP Response status:', response.status);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to send OTP' }));
      console.error('OTP Error:', error);
      throw new Error(error.detail || 'Failed to send OTP');
    }

    const data = await response.json();
    console.log('OTP Success:', data);
    return data;
  } catch (err) {
    console.error('OTP Request failed:', err);
    throw err;
  }
}

/**
 * Verify OTP
 */
export async function verifyOtp(
  phone: string,
  otp: string,
  purpose: string = 'registration'
): Promise<{ success: boolean; message: string; verification_token?: string }> {
  const response = await fetch(`${API_BASE_URL}/auth/verify-otp`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone, otp, purpose }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'OTP verification failed' }));
    throw new Error(error.detail || 'OTP verification failed');
  }

  return response.json();
}

/**
 * Register new user
 */
export async function register(data: {
  full_name: string;
  phone: string;
  verification_token: string;
  email?: string;
  state_or_ut?: string;
  city?: string;
}): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(error.detail || 'Registration failed');
  }

  return response.json();
}

/**
 * Login user
 */
export async function login(phone: string, verification_token: string): Promise<AuthResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // Render free tier cold starts can exceed 15s

  try {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ phone, verification_token }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Invalid credentials' }));
      throw new Error(error.detail || 'Login failed');
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Login request timed out. Please try again.');
    }
    throw error;
  }
}

/**
 * Refresh access token
 */
export async function refreshToken(
  refresh_token: string
): Promise<{ access_token: string; token_type: string; expires_in: number }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Get current user info
 */
export async function getCurrentUser(access_token: string): Promise<User> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${access_token}`,
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error('Failed to get user info');
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Logout user
 */
export async function logout(access_token: string): Promise<void> {
  await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${access_token}`,
    },
  });
}

