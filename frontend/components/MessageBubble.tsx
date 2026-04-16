'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { Message, Source } from '@/types';

function topicLabel(code: string | null | undefined): string {
  if (!code) return '';
  return code.replace(/_/g, ' ');
}
import { User, GraduationCap, ChevronDown, ChevronUp, FileText, AlertCircle, CheckCircle2, BookOpen } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
  onSuggestedReply: (reply: string) => void;
}

/** True when the model likely refused or said it could not answer (avoid "Verified" in that case). */
/** Fix common LLM mistakes so react-markdown can parse (headings/lists glued to prior text). */
function normalizeMarkdownForRender(content: string): string {
  let s = content;
  if (!s.trim()) return s;
  // "### Heading" or "## H" immediately after non-newline text → insert breaks
  s = s.replace(/([^\n#])(#{1,6}\s)/g, '$1\n\n$2');
  // "1." "2." after a sentence on the same line
  s = s.replace(/([.!?])\s+(\d{1,2}\.\s)/g, '$1\n\n$2');
  // Drop accidental standalone pipe lines that break GFM table parsing.
  s = s.replace(/^\|\s*$/gm, '');
  // Ensure table header starts on a clean line.
  s = s.replace(/([^\n])\n(\|[^\n]+\|\n\|[-:| ]+\|)/g, '$1\n\n$2');
  return s;
}

function hasMarkdownTable(content: string): boolean {
  return /\|[^\n]+\|\s*\n\|[\-:\s|]+\|/m.test(content);
}

type ParsedCutoffTable = {
  rows: Array<{
    idx: string;
    institution: string;
    state: string;
    category: string;
    quota: string;
    domicile: string;
    air: string;
    score: string;
    round: string;
  }>;
  contentWithoutTable: string;
};

function parseCutoffTable(content: string): ParsedCutoffTable | null {
  const start = content.indexOf('| # | Institution | State | Category | Quota | Domicile | AIR | Score | Round |');
  if (start < 0) return null;

  let end = content.length;
  const disclaimerIdx = content.indexOf('\n> Disclaimer', start);
  const ctaIdx = content.indexOf('\nWould you like', start);
  if (disclaimerIdx >= 0) end = Math.min(end, disclaimerIdx);
  if (ctaIdx >= 0) end = Math.min(end, ctaIdx);

  const tableBlock = content.slice(start, end);
  const firstRowMatch = tableBlock.match(/\|\s*\d+\s*\|/);
  if (!firstRowMatch || firstRowMatch.index === undefined) return null;

  const rowsArea = tableBlock.slice(firstRowMatch.index);
  const tokens = rowsArea
    .split('|')
    .map((t) => t.trim())
    .filter((t) => t.length > 0);

  const rows: ParsedCutoffTable['rows'] = [];
  for (let i = 0; i < tokens.length; ) {
    if (!/^\d+$/.test(tokens[i])) {
      i += 1;
      continue;
    }
    if (i + 8 >= tokens.length) break;
    rows.push({
      idx: tokens[i],
      institution: tokens[i + 1],
      state: tokens[i + 2],
      category: tokens[i + 3],
      quota: tokens[i + 4],
      domicile: tokens[i + 5],
      air: tokens[i + 6],
      score: tokens[i + 7],
      round: tokens[i + 8],
    });
    i += 9;
  }

  if (rows.length === 0) return null;
  const contentWithoutTable = `${content.slice(0, start)}${content.slice(end)}`.trim();
  return { rows, contentWithoutTable };
}

function isLikelyRefusalOrNoInfo(content: string): boolean {
  const t = content.toLowerCase();
  if (t.includes('not available')) return true;
  if (t.includes('could not find') || t.includes("couldn't find")) return true;
  if (t.includes('no information') && t.includes('sorry')) return true;
  if (t.includes("don't have") && (t.includes('information') || t.includes('details'))) return true;
  if (t.includes('unable to find') || t.includes('cannot answer')) return true;
  return false;
}

export default function MessageBubble({ message, onSuggestedReply }: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const parsedCutoffTable = parseCutoffTable(message.content || '');
  const contentForMarkdown = parsedCutoffTable ? parsedCutoffTable.contentWithoutTable : (message.content || '');
  const normalizedContent = normalizeMarkdownForRender(contentForMarkdown);
  const containsTable = hasMarkdownTable(normalizedContent);

  // Don't render empty assistant messages (they're being streamed)
  if (!isUser && (!message.content || message.content.trim() === '') && !message.needsClarification) {
    return null;
  }

  return (
    <div className={`flex items-start gap-4 message-enter ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${
          isUser
            ? 'bg-gradient-to-br from-gray-700 to-gray-800'
            : message.isError
            ? 'bg-gradient-to-br from-red-500 to-red-600'
            : 'bg-gradient-to-br from-blue-600 to-indigo-600'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : message.isError ? (
          <AlertCircle className="w-5 h-5 text-white" />
        ) : (
          <GraduationCap className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        {/* Role label */}
        <p className={`text-xs font-semibold mb-1.5 ${isUser ? 'text-gray-500 dark:text-gray-400' : 'text-blue-600 dark:text-blue-400'}`}>
          {isUser ? 'You' : 'Med Buddy'}
        </p>
        
        <div
          className={`inline-block rounded-2xl p-4 shadow-sm ${
            isUser
              ? 'bg-gradient-to-br from-gray-700 to-gray-800 text-white rounded-tr-sm'
              : message.isError
              ? 'bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30 text-red-800 dark:text-red-300 rounded-tl-sm'
              : 'bg-white dark:bg-slate-800 border border-gray-100 dark:border-slate-700 text-gray-800 dark:text-gray-200 rounded-tl-sm shadow-md dark:shadow-lg dark:shadow-black/10'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="assistant-markdown prose prose-sm max-w-none prose-headings:text-gray-800 dark:prose-headings:text-white prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-li:text-gray-700 dark:prose-li:text-gray-300 prose-strong:text-gray-800 dark:prose-strong:text-white">
              <ReactMarkdown
                remarkPlugins={containsTable ? [remarkGfm] : [remarkGfm, remarkBreaks]}
                urlTransform={(url) => url}
                components={{
                  table: ({ children }) => (
                    <div className="my-4 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-600">
                      <table className="min-w-full border-collapse text-sm">{children}</table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-slate-100 dark:bg-slate-700">{children}</thead>
                  ),
                  tbody: ({ children }) => (
                    <tbody className="bg-white dark:bg-slate-800">{children}</tbody>
                  ),
                  tr: ({ children }) => (
                    <tr className="border-b border-slate-200 dark:border-slate-600">{children}</tr>
                  ),
                  th: ({ children }) => (
                    <th className="px-3 py-2 text-left font-semibold text-slate-800 dark:text-slate-100">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-3 py-2 text-slate-700 dark:text-slate-200 align-top">
                      {children}
                    </td>
                  ),
                }}
              >
                {normalizedContent}
              </ReactMarkdown>

              {parsedCutoffTable && (
                <div className="my-4 overflow-x-auto rounded-xl border border-slate-600 bg-slate-900/80">
                  <table className="min-w-full border-collapse text-sm">
                    <thead className="bg-slate-800">
                      <tr className="border-b border-slate-600">
                        {['Institution Name', 'State', 'Category', 'Quota', 'Domicile', 'AIR', 'Score', 'Round'].map((h) => (
                          <th key={h} className="px-3 py-2 text-left font-semibold text-slate-100 whitespace-nowrap">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {parsedCutoffTable.rows.map((row) => (
                        <tr key={`${row.idx}-${row.institution}`} className="border-b border-slate-700">
                          <td className="px-3 py-2 text-slate-100 font-medium">{row.institution}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.state}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.category}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.quota}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.domicile}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.air}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.score}</td>
                          <td className="px-3 py-2 text-slate-200 whitespace-nowrap">{row.round}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Source indicator for assistant - only show when there's actual content and not clarification */}
        {!isUser && !message.isError && !message.needsClarification && message.content && message.content.trim() !== '' && (
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            {isLikelyRefusalOrNoInfo(message.content) ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 rounded-full">
                <BookOpen className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400" />
                <span className="text-xs font-medium text-amber-800 dark:text-amber-300">
                  References from official PDFs — excerpts may not list every detail you asked for
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/30 rounded-full">
                <CheckCircle2 className="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
                <span className="text-xs font-medium text-green-700 dark:text-green-400">Verified from Official Document</span>
              </div>
            )}
          </div>
        )}

        {/* Sources section - only show when there's actual content and not clarification */}
        {!isUser && !message.needsClarification && message.content && message.content.trim() !== '' && message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors font-medium"
            >
              <BookOpen className="w-4 h-4" />
              <span>View {message.sources.length} Reference(s) from NEET Bulletin</span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showSources && (
              <div className="mt-3 space-y-2">
                {message.sources.map((source, index) => (
                  <SourceCard key={index} source={source} index={index} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Suggested replies (guided chips) */}
        {!isUser && message.suggestedReplies && message.suggestedReplies.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.suggestedReplies.slice(0, 6).map((reply, idx) => (
              <button
                key={`${reply}-${idx}`}
                onClick={() => onSuggestedReply(reply)}
                className="px-3 py-1.5 rounded-full border border-blue-200 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-medium hover:bg-blue-100 dark:hover:bg-blue-800/40 transition-colors"
              >
                {reply}
              </button>
            ))}
          </div>
        )}

        {/* Timestamp - only show when there's content */}
        {message.content && message.content.trim() !== '' && (
          <p className={`text-xs text-gray-400 dark:text-gray-500 mt-2 ${isUser ? 'text-right' : ''}`}>
            {formatTime(message.timestamp)}
          </p>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-500/30 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="p-1.5 bg-blue-100 dark:bg-blue-500/20 rounded-lg">
          <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400" />
        </div>
        <span className="text-sm font-semibold text-gray-800 dark:text-white">
          {source.file_name}
        </span>
        {source.page && (
          <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-300 text-xs font-medium rounded-full">
            Page {source.page}
          </span>
        )}
      </div>
      {(source.doc_topic || source.chunk_category) && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {source.doc_topic && (
            <span
              className="px-2 py-0.5 bg-violet-100 dark:bg-violet-500/20 text-violet-800 dark:text-violet-200 text-xs rounded-full"
              title="Document scope (set when uploading)"
            >
              Doc: {topicLabel(source.doc_topic)}
            </span>
          )}
          {source.chunk_category && (
            <span
              className="px-2 py-0.5 bg-slate-200 dark:bg-slate-600/40 text-slate-800 dark:text-slate-200 text-xs rounded-full"
              title="Chunk topic (AI, page-wise)"
            >
              Chunk: {topicLabel(source.chunk_category)}
            </span>
          )}
        </div>
      )}
      <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed line-clamp-3">{source.text_snippet}</p>
    </div>
  );
}

function formatTime(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}
