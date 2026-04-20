'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export default function ThemeToggle({ language = 'en' }: { language?: 'en' | 'hi' | 'mr' }) {
  const { theme, toggleTheme } = useTheme();
  const title =
    language === 'hi'
      ? theme === 'light'
        ? 'डार्क मोड पर स्विच करें'
        : 'लाइट मोड पर स्विच करें'
      : language === 'mr'
      ? theme === 'light'
        ? 'डार्क मोडवर स्विच करा'
        : 'लाईट मोडवर स्विच करा'
      : theme === 'light'
      ? 'Switch to dark mode'
      : 'Switch to light mode';

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg transition-colors bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600"
      title={title}
    >
      {theme === 'light' ? (
        <Moon className="w-5 h-5 text-gray-600" />
      ) : (
        <Sun className="w-5 h-5 text-yellow-400" />
      )}
    </button>
  );
}
