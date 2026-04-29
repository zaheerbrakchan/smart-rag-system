'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { Message, Source } from '@/types';

function topicLabel(code: string | null | undefined): string {
  if (!code) return '';
  return code.replace(/_/g, ' ');
}
import { User, GraduationCap, ChevronDown, ChevronUp, FileText, AlertCircle, CheckCircle2, BookOpen, Search } from 'lucide-react';
import { getCutoffProfileOptions } from '@/services/api';

interface MessageBubbleProps {
  message: Message;
  onSuggestedReply: (reply: string) => void;
  onCutoffProfileSubmit: (payload: { state: string; category: string; subCategory?: string }) => void;
  language: 'en' | 'hi' | 'mr';
  referencesEnabledGlobal?: boolean;
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
  contentBeforeTable: string;
  contentAfterTable: string;
};

function parseCutoffTable(content: string): ParsedCutoffTable | null {
  const headerMatch = content.match(/\|\s*#\s*\|\s*Institution\s*\|\s*State\s*\|[^\n]*Round\s*\|/i);
  if (!headerMatch || headerMatch.index === undefined) return null;
  const headerLine = headerMatch[0];
  const start = headerMatch.index;
  const hasTypeColumn = /\|\s*Type\s*\|/i.test(headerLine);

  // Find true markdown-table boundary first so any in-between loader/progress text
  // (between table and "Quick Interpretation") is preserved in contentAfterTable.
  const tail = content.slice(start);
  const tailLines = tail.split('\n');
  let tableCharLen = 0;
  for (const line of tailLines) {
    const trimmed = line.trim();
    const isTableLine = trimmed.length > 0 && trimmed.startsWith('|') && trimmed.endsWith('|');
    if (!isTableLine) break;
    tableCharLen += line.length + 1; // include newline
  }

  let end = start + tableCharLen;
  if (end <= start) end = content.length;

  const disclaimerIdxFromNote = content.indexOf('\n> *Note', start);
  const disclaimerIdxLegacy = content.indexOf('\n> Disclaimer', start);
  const disclaimerIdx =
    disclaimerIdxFromNote >= 0 && disclaimerIdxLegacy >= 0
      ? Math.min(disclaimerIdxFromNote, disclaimerIdxLegacy)
      : Math.max(disclaimerIdxFromNote, disclaimerIdxLegacy);
  const ctaIdx = content.indexOf('\nWould you like', start);
  // Keep only hard stop guards that should end table parsing if they somehow appear
  // before computed table end.
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
    const rowWidth = hasTypeColumn ? 10 : 9;
    if (i + rowWidth - 1 >= tokens.length) break;

    const categoryIdx = hasTypeColumn ? i + 4 : i + 3;
    const quotaIdx = hasTypeColumn ? i + 5 : i + 4;
    const domicileIdx = hasTypeColumn ? i + 6 : i + 5;
    const airIdx = hasTypeColumn ? i + 7 : i + 6;
    const scoreIdx = hasTypeColumn ? i + 8 : i + 7;
    const roundIdx = hasTypeColumn ? i + 9 : i + 8;

    rows.push({
      idx: tokens[i],
      institution: tokens[i + 1],
      state: tokens[i + 2],
      category: tokens[categoryIdx],
      quota: tokens[quotaIdx],
      domicile: tokens[domicileIdx],
      air: tokens[airIdx],
      score: tokens[scoreIdx],
      round: tokens[roundIdx],
    });
    i += rowWidth;
  }

  if (rows.length === 0) return null;
  const contentBeforeTable = content.slice(0, start).trim();
  const contentAfterTable = content.slice(end).trim();
  return { rows, contentBeforeTable, contentAfterTable };
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

function hasExternalWebSource(sources?: Source[]): boolean {
  if (!sources || sources.length === 0) return false;
  return sources.some((s) => {
    const dtype = String(s.document_type || '').toLowerCase();
    if (dtype === 'web_search') return true;
    if (dtype.includes('web')) return true;
    const name = String(s.file_name || '').toLowerCase();
    return name.startsWith('http://') || name.startsWith('https://');
  });
}

export default function MessageBubble({
  message,
  onSuggestedReply,
  onCutoffProfileSubmit,
  language,
  referencesEnabledGlobal = true,
}: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);
  const [selectedState, setSelectedState] = useState(message.cutoffProfileForm?.selectedState || '');
  const [selectedCategory, setSelectedCategory] = useState(message.cutoffProfileForm?.selectedCategory || '');
  const [selectedSubCategory, setSelectedSubCategory] = useState(message.cutoffProfileForm?.selectedSubCategory || 'NOT_SURE');
  const [categoryOptions, setCategoryOptions] = useState<string[]>(message.cutoffProfileForm?.categories || []);
  const [subCategoryOptions, setSubCategoryOptions] = useState<string[]>(message.cutoffProfileForm?.subCategories || []);
  const [profileSubmitting, setProfileSubmitting] = useState(false);
  const isUser = message.role === 'user';
  const youLabel = language === 'hi' ? 'आप' : language === 'mr' ? 'तुम्ही' : 'You';
  const buddyLabel = 'Med Buddy';
  const tableHeaders =
    language === 'hi'
      ? ['संस्थान', 'राज्य', 'श्रेणी', 'कोटा', 'निवास', 'AIR', 'स्कोर', 'राउंड']
      : language === 'mr'
      ? ['संस्था', 'राज्य', 'प्रवर्ग', 'कोटा', 'निवास', 'AIR', 'स्कोर', 'राउंड']
      : ['Institution Name', 'State', 'Category', 'Quota', 'Domicile', 'AIR', 'Score', 'Round'];
  const verifiedLabel =
    language === 'hi'
      ? 'आधिकारिक दस्तावेज़ से सत्यापित'
      : language === 'mr'
      ? 'अधिकृत दस्तऐवजातून पडताळलेले'
      : 'Verified from Official Document';
  const externalVerifiedLabel = 'Verified from external sources; user discretion advised';
  const viewRefsLabel =
    language === 'hi'
      ? `NEET बुलेटिन से ${message.sources?.length || 0} संदर्भ देखें`
      : language === 'mr'
      ? `NEET बुलेटिनमधील ${message.sources?.length || 0} संदर्भ पहा`
      : `View ${message.sources?.length || 0} Reference(s) from NEET Bulletin`;
  const parsedCutoffTable = parseCutoffTable(message.content || '');
  const contentBeforeTable = parsedCutoffTable ? parsedCutoffTable.contentBeforeTable : (message.content || '');
  const contentAfterTable = parsedCutoffTable ? parsedCutoffTable.contentAfterTable : '';
  const normalizedContent = normalizeMarkdownForRender(contentBeforeTable);
  const normalizedAfterContent = normalizeMarkdownForRender(contentAfterTable);
  const containsTable = hasMarkdownTable(normalizedContent);
  const isExternalAnswer = message.sourceOrigin === 'web' || hasExternalWebSource(message.sources);
  const isOfficialAnswer = message.sourceOrigin === 'kb' || ((message.sources?.length || 0) > 0 && !isExternalAnswer);
  const referencesVisible = referencesEnabledGlobal && message.referencesEnabled !== false;
  const shouldShowVerificationBadge =
    !isUser &&
    !message.isError &&
    !message.needsClarification &&
    Boolean(message.content && message.content.trim() !== '') &&
    !isLikelyRefusalOrNoInfo(message.content) &&
    (isExternalAnswer || isOfficialAnswer);
  const hasCutoffProfileForm = !isUser && !!message.cutoffProfileForm;
  const groupedCutoffRows = parsedCutoffTable
    ? parsedCutoffTable.rows.reduce<Record<string, ParsedCutoffTable['rows']>>((acc, row) => {
        const key = row.state || 'Unknown';
        if (!acc[key]) acc[key] = [];
        acc[key].push(row);
        return acc;
      }, {})
    : {};
  const cutoffStates = Object.keys(groupedCutoffRows);
  const isMultiStateCutoffTable = cutoffStates.length > 1;
  const tableHeadersWithoutState =
    language === 'hi'
      ? ['संस्थान', 'श्रेणी', 'कोटा', 'निवास', 'AIR', 'स्कोर', 'राउंड']
      : language === 'mr'
      ? ['संस्था', 'प्रवर्ग', 'कोटा', 'निवास', 'AIR', 'स्कोर', 'राउंड']
      : ['Institution Name', 'Category', 'Quota', 'Domicile', 'AIR', 'Score', 'Round'];
  const stateOptions = (message.cutoffProfileForm?.states || []).filter(
    (st) => String(st).toUpperCase() !== 'MCC'
  );

  useEffect(() => {
    setSelectedState(message.cutoffProfileForm?.selectedState || '');
    setSelectedCategory(message.cutoffProfileForm?.selectedCategory || '');
    setSelectedSubCategory(message.cutoffProfileForm?.selectedSubCategory || 'NOT_SURE');
    setCategoryOptions(message.cutoffProfileForm?.categories || []);
    setSubCategoryOptions(message.cutoffProfileForm?.subCategories || []);
  }, [message.cutoffProfileForm]);

  useEffect(() => {
    if (!hasCutoffProfileForm || !selectedState || selectedState === 'NOT_SURE') {
      setCategoryOptions([]);
      return;
    }
    let cancelled = false;
    const loadCategories = async () => {
      try {
        const data = await getCutoffProfileOptions(selectedState);
        if (!cancelled) {
          setCategoryOptions(data.categories || []);
        }
      } catch {
        // Keep existing options on transient failures.
      }
    };
    void loadCategories();
    return () => {
      cancelled = true;
    };
  }, [hasCutoffProfileForm, selectedState]);

  useEffect(() => {
    if (
      !hasCutoffProfileForm
      || !selectedState
      || selectedState === 'NOT_SURE'
      || !selectedCategory
      || selectedCategory === 'NOT_SURE'
    ) {
      setSubCategoryOptions([]);
      setSelectedSubCategory('NOT_SURE');
      return;
    }
    let cancelled = false;
    const loadSubCategories = async () => {
      try {
        const data = await getCutoffProfileOptions(selectedState, selectedCategory);
        if (!cancelled) {
          setSubCategoryOptions(data.sub_categories || []);
        }
      } catch {
        if (!cancelled) {
          setSubCategoryOptions([]);
        }
      }
    };
    void loadSubCategories();
    return () => {
      cancelled = true;
    };
  }, [hasCutoffProfileForm, selectedState, selectedCategory]);

  // Don't render empty assistant messages (they're being streamed)
  if (!isUser && (!message.content || message.content.trim() === '') && !message.needsClarification) {
    return null;
  }

  return (
    <div className={`flex items-start gap-2 md:gap-4 message-enter min-w-0 ${isUser ? 'flex-row-reverse' : ''}`}>
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
      <div className={`flex-1 min-w-0 max-w-full md:max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        {/* Role label */}
        <p className={`text-xs font-semibold mb-1.5 ${isUser ? 'text-gray-500 dark:text-gray-400' : 'text-blue-600 dark:text-blue-400'}`}>
          {isUser ? youLabel : buddyLabel}
        </p>
        
        <div
          className={`${isUser ? 'inline-block' : 'block w-full'} max-w-full rounded-2xl p-3 md:p-4 shadow-sm ${
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
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 my-3 italic text-slate-600 dark:text-slate-400 text-sm leading-relaxed">
                      {children}
                    </blockquote>
                  ),
                  table: ({ children }) => (
                    <div className="my-4 max-w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-600">
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

              {parsedCutoffTable && !isMultiStateCutoffTable && (
                <div className="my-4 max-w-full overflow-x-auto rounded-xl border border-slate-600 bg-slate-900/80">
                  <table className="min-w-full border-collapse text-sm">
                    <thead className="bg-slate-800">
                      <tr className="border-b border-slate-600">
                        {tableHeaders.map((h) => (
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

              {parsedCutoffTable && isMultiStateCutoffTable && (
                <div className="my-4 space-y-4">
                  {cutoffStates.map((stateName) => {
                    const stateRows = groupedCutoffRows[stateName] || [];
                    return (
                      <div key={stateName} className="max-w-full rounded-xl border border-slate-600 bg-slate-900/80 overflow-x-auto">
                        <div className="px-3 py-2 bg-slate-800/90 border-b border-slate-600">
                          <span className="inline-flex items-center rounded-full bg-blue-600/20 text-blue-300 border border-blue-500/40 px-2.5 py-1 text-xs font-semibold">
                            {stateName}
                          </span>
                        </div>
                        <table className="min-w-full border-collapse text-sm">
                          <thead className="bg-slate-800">
                            <tr className="border-b border-slate-600">
                              {tableHeadersWithoutState.map((h) => (
                                <th key={`${stateName}-${h}`} className="px-3 py-2 text-left font-semibold text-slate-100 whitespace-nowrap">
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {stateRows.map((row) => (
                              <tr key={`${stateName}-${row.idx}-${row.institution}`} className="border-b border-slate-700">
                                <td className="px-3 py-2 text-slate-100 font-medium">{row.institution}</td>
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
                    );
                  })}
                </div>
              )}

              {parsedCutoffTable && message.cutoffInterpretationLoading && (
                <div className="my-3 flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
                  <Search className="w-4 h-4 text-blue-500 dark:text-blue-400 animate-pulse" />
                  <span>Generating quick interpretation...</span>
                  <div className="flex gap-1">
                    <span className="typing-dot w-2 h-2 bg-blue-400 rounded-full"></span>
                    <span className="typing-dot w-2 h-2 bg-blue-400 rounded-full"></span>
                    <span className="typing-dot w-2 h-2 bg-blue-400 rounded-full"></span>
                  </div>
                  <span>Thinking...</span>
                </div>
              )}

              {parsedCutoffTable && normalizedAfterContent && (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks]}
                  urlTransform={(url) => url}
                  components={{
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 my-3 italic text-slate-600 dark:text-slate-400 text-sm leading-relaxed">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {normalizedAfterContent}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>

        {/* Source indicator for assistant - only show when there's actual content and not clarification */}
        {shouldShowVerificationBadge && (
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            {isExternalAnswer ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-violet-50 dark:bg-violet-500/10 border border-violet-200 dark:border-violet-500/30 rounded-full">
                <CheckCircle2 className="w-3.5 h-3.5 text-violet-700 dark:text-violet-300" />
                <span className="text-xs font-medium text-violet-700 dark:text-violet-300">{externalVerifiedLabel}</span>
              </div>
            ) : isOfficialAnswer ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/30 rounded-full">
                <CheckCircle2 className="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
                <span className="text-xs font-medium text-green-700 dark:text-green-400">{verifiedLabel}</span>
              </div>
            ) : null}
          </div>
        )}

        {/* Sources section - only show when there's actual content and not clarification */}
        {referencesVisible && !isUser && !message.needsClarification && message.content && message.content.trim() !== '' && message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors font-medium"
            >
              <BookOpen className="w-4 h-4" />
              <span>{viewRefsLabel}</span>
              {showSources ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showSources && (
              <div className="mt-3 space-y-2">
                {message.sources.map((source, index) => (
                  <SourceCard key={index} source={source} index={index} language={language} />
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

        {/* Cutoff profile form */}
        {hasCutoffProfileForm && (
          <div className="mt-3 rounded-xl border border-blue-200 dark:border-blue-500/70 bg-blue-50/70 dark:bg-slate-900 p-3 space-y-3 shadow-sm dark:shadow-lg dark:shadow-black/20">
            <div>
              <label className="text-xs font-semibold text-gray-700 dark:text-slate-100">Home state</label>
              <select
                className="mt-1 w-full rounded-lg border border-gray-300 dark:border-slate-400 bg-white dark:bg-slate-800 px-3 py-2 text-sm text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/70 cursor-pointer disabled:cursor-not-allowed"
                value={selectedState}
                onChange={(e) => {
                  setSelectedState(e.target.value);
                  setSelectedCategory('');
                  setSelectedSubCategory('NOT_SURE');
                }}
              >
                <option value="">Select state</option>
                {stateOptions.map((st) => (
                  <option key={st} value={st}>{st}</option>
                ))}
                <option value="NOT_SURE">Not sure</option>
              </select>
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-700 dark:text-slate-100">Category</label>
              <select
                className="mt-1 w-full rounded-lg border border-gray-300 dark:border-slate-400 bg-white dark:bg-slate-800 px-3 py-2 text-sm text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/70 cursor-pointer disabled:cursor-not-allowed"
                value={selectedCategory}
                onChange={(e) => {
                  setSelectedCategory(e.target.value);
                  setSelectedSubCategory('NOT_SURE');
                }}
                disabled={!selectedState}
              >
                <option value="">Select category</option>
                <option value="NOT_SURE">Not sure</option>
                {categoryOptions.map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-700 dark:text-slate-100">Sub-category (optional)</label>
              <select
                className="mt-1 w-full rounded-lg border border-gray-300 dark:border-slate-400 bg-white dark:bg-slate-800 px-3 py-2 text-sm text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/70 cursor-pointer disabled:cursor-not-allowed"
                value={selectedSubCategory}
                onChange={(e) => setSelectedSubCategory(e.target.value)}
                disabled={!selectedState || !selectedCategory || selectedCategory === 'NOT_SURE'}
              >
                <option value="NOT_SURE">Not sure</option>
                {subCategoryOptions.map((sub) => (
                  <option key={sub} value={sub}>{sub}</option>
                ))}
              </select>
            </div>
            <button
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              disabled={!selectedState || !selectedCategory || profileSubmitting}
              onClick={async () => {
                setProfileSubmitting(true);
                try {
                  onCutoffProfileSubmit({
                    state: selectedState,
                    category: selectedCategory,
                    subCategory: selectedSubCategory !== 'NOT_SURE' ? selectedSubCategory : undefined,
                  });
                } finally {
                  setProfileSubmitting(false);
                }
              }}
            >
              {profileSubmitting ? 'Submitting...' : 'Submit details'}
            </button>
          </div>
        )}

        {/* Timestamp - only show when there's content */}
        {message.content && message.content.trim() !== '' && (
          <p className={`text-xs text-gray-400 dark:text-gray-500 mt-2 ${isUser ? 'text-right' : ''}`}>
            {formatTime(message.timestamp, language)}
          </p>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source, index, language }: { source: Source; index: number; language: 'en' | 'hi' | 'mr' }) {
  const pageLabel = language === 'hi' ? 'पेज' : language === 'mr' ? 'पृष्ठ' : 'Page';
  const docLabel = language === 'hi' ? 'दस्तावेज़' : language === 'mr' ? 'दस्तऐवज' : 'Doc';
  const chunkLabel = language === 'hi' ? 'खंड' : language === 'mr' ? 'भाग' : 'Chunk';
  const docTooltip =
    language === 'hi'
      ? 'दस्तावेज़ का दायरा (अपलोड के समय सेट)'
      : language === 'mr'
      ? 'दस्तऐवजाचा व्याप्ती (अपलोडवेळी सेट)'
      : 'Document scope (set when uploading)';
  const chunkTooltip =
    language === 'hi'
      ? 'खंड विषय (AI, पेज-वार)'
      : language === 'mr'
      ? 'भाग विषय (AI, पृष्ठानुसार)'
      : 'Chunk topic (AI, page-wise)';
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
            {pageLabel} {source.page}
          </span>
        )}
      </div>
      {(source.doc_topic || source.chunk_category) && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {source.doc_topic && (
            <span
              className="px-2 py-0.5 bg-violet-100 dark:bg-violet-500/20 text-violet-800 dark:text-violet-200 text-xs rounded-full"
              title={docTooltip}
            >
              {docLabel}: {topicLabel(source.doc_topic)}
            </span>
          )}
          {source.chunk_category && (
            <span
              className="px-2 py-0.5 bg-slate-200 dark:bg-slate-600/40 text-slate-800 dark:text-slate-200 text-xs rounded-full"
              title={chunkTooltip}
            >
              {chunkLabel}: {topicLabel(source.chunk_category)}
            </span>
          )}
        </div>
      )}
      <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed line-clamp-3">{source.text_snippet}</p>
    </div>
  );
}

function formatTime(date: Date, language: 'en' | 'hi' | 'mr'): string {
  const locale = language === 'hi' ? 'hi-IN' : language === 'mr' ? 'mr-IN' : 'en-US';
  return new Intl.DateTimeFormat(locale, {
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}
