import type { Metadata } from 'next';
import './globals.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  title: 'NEET UG 2026 Assistant | Get My University',
  description: 'Your AI-powered assistant for NEET UG 2026 queries. Get instant, accurate answers from the official NTA Information Bulletin.',
  keywords: 'NEET UG 2026, NTA, NEET eligibility, NEET exam dates, medical entrance, Get My University',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
