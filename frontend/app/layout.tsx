import type { Metadata } from 'next';
import Script from 'next/script';
import './globals.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  title: 'NEET UG 2026 Med Assist | Get My University',
  description:
    'Your AI-powered assistant for NEET UG 2026 queries. Get instant, accurate answers from the official NTA Information Bulletin.',
  keywords:
    'NEET UG 2026, NTA, NEET eligibility, NEET exam dates, medical entrance, Get My University, Med Assist',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="light" suppressHydrationWarning>
      <body className="bg-gray-50 antialiased">
        <Script id="theme-init" strategy="beforeInteractive">
          {`(function(){try{var k='app_theme_v2';var t=localStorage.getItem(k);var c='light';if(t==='dark')c='dark';else if(t==='light')c='light';document.documentElement.classList.remove('light','dark');document.documentElement.classList.add(c);}catch(e){document.documentElement.classList.remove('light','dark');document.documentElement.classList.add('light');}})();`}
        </Script>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
