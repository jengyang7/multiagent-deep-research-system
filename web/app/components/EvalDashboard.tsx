'use client';

import { useEffect, useMemo, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types (mirror api/main.py response shapes + eval/schema.py)
// ---------------------------------------------------------------------------

interface RunStats {
  lead_model?: string;
  subagent_model?: string;
  input_tokens?: number;
  output_tokens?: number;
  cached_tokens?: number;
  total_tokens?: number;
  cost_usd?: number;
  elapsed_seconds?: number;
}

interface RunSummary {
  id: string;
  query: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  stats: RunStats | null;
}

interface ModelOption {
  id: string;
  label?: string;
  description?: string;
}

interface GlobalSummary {
  runs_evaluated: number;
  pass_rate: number | null;
  grounding_rate: number | null;
  faithfulness_rate: number | null;
  completeness_rate: number | null;
  relevance_score: number | null;
}

interface EvalReportSummary {
  id: string;
  run_id: string;
  query: string;
  generated_at: string;
  passed: boolean;
  total_findings: number;
  ungrounded_count: number;
  total_citations: number;
  unfaithful_count: number;
  uncited_count: number;
  failure_reasons: string[];
  eval_model: string;
  eval_cost_usd: number;
  recall_score: number;
  relevance_score: number;
}

interface GroundingResult {
  subtask: string;
  claim: string;
  evidence_span: string;
  citation_url: string;
  grounded: boolean;
  similarity: number;
  method: string;
  fetch_chars: number;
  note?: string | null;
}

interface FaithfulnessVerdict {
  citation_index: number;
  report_sentence: string;
  matched_finding_claims: string[];
  faithful: boolean;
  confidence: number;
  reasoning: string;
}

interface UncitedSentence {
  sentence: string;
  section: string;
}

interface SubtopicCoverage {
  subtopic: string;
  covered: boolean;
  note: string;
}

interface CompletenessResult {
  subtopics: SubtopicCoverage[];
  recall_score: number;
}

interface RelevanceResult {
  score: number;
  reasoning: string;
}

interface EvalReportDetail extends EvalReportSummary {
  report: {
    grounding_results: GroundingResult[];
    faithfulness_results: FaithfulnessVerdict[];
    uncited_sentences: UncitedSentence[];
    completeness: CompletenessResult;
    relevance: RelevanceResult;
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groundingRate(r: EvalReportSummary): number | null {
  if (r.total_findings === 0) return null;
  return ((r.total_findings - r.ungrounded_count) / r.total_findings) * 100;
}

function faithfulnessRate(r: EvalReportSummary): number | null {
  if (r.total_citations === 0) return null;
  return ((r.total_citations - r.unfaithful_count) / r.total_citations) * 100;
}

function fmtPct(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(0)}%`;
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

function fmtCost(v?: number): string {
  return v === undefined ? '—' : `$${v.toFixed(4)}`;
}

function fmtScore(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(1)}/5.0`;
}

// ---------------------------------------------------------------------------
// Small icons (inline, matching page.tsx conventions)
// ---------------------------------------------------------------------------

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-blue-500 flex-shrink-0" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-red-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Trend chart — lightweight inline SVG, no charting dependency
// ---------------------------------------------------------------------------

function TrendChart({
  reports, selectedId, onSelect,
}: {
  reports: EvalReportSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (reports.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-gray-400">
        No eval runs yet — run an eval from the table below to see trends.
      </div>
    );
  }

  const W = 720;
  const H = 200;
  const padL = 36;
  const padR = 12;
  const padT = 12;
  const padB = 24;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const n = reports.length;
  const xFor = (i: number) => (n === 1 ? padL + innerW / 2 : padL + (i / (n - 1)) * innerW);
  const yFor = (pct: number) => padT + innerH - (pct / 100) * innerH;

  const groundingPts = reports.map((r, i) => ({ r, x: xFor(i), pct: groundingRate(r) }));
  const faithPts = reports.map((r, i) => ({ r, x: xFor(i), pct: faithfulnessRate(r) }));

  const linePath = (pts: { x: number; pct: number | null }[]) =>
    pts
      .filter(p => p.pct !== null)
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${yFor(p.pct as number)}`)
      .join(' ');

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-48" preserveAspectRatio="none">
        {/* gridlines + y-axis labels at 0/50/100% */}
        {[0, 50, 100].map(pct => (
          <g key={pct}>
            <line x1={padL} y1={yFor(pct)} x2={W - padR} y2={yFor(pct)} stroke="#e5e7eb" strokeWidth={1} />
            <text x={padL - 6} y={yFor(pct) + 3} textAnchor="end" fontSize="9" fill="#9ca3af">{pct}%</text>
          </g>
        ))}

        {/* grounding line (blue, solid) */}
        <path d={linePath(groundingPts)} fill="none" stroke="#3b82f6" strokeWidth={2} />
        {/* faithfulness line (emerald, dashed) — distinguishes the two lines when their values overlap */}
        <path d={linePath(faithPts)} fill="none" stroke="#10b981" strokeWidth={2} strokeDasharray="5 4" />

        {/* points (clickable) */}
        {reports.map((r, i) => {
          const gPct = groundingRate(r);
          const fPct = faithfulnessRate(r);
          const isSelected = r.id === selectedId;
          return (
            <g key={r.id} className="cursor-pointer" onClick={() => onSelect(r.id)}>
              {/* invisible wide hit-target */}
              <rect x={xFor(i) - (innerW / Math.max(n - 1, 1)) / 2} y={padT} width={innerW / Math.max(n, 1)} height={innerH} fill="transparent" />
              {gPct !== null && (
                <circle cx={xFor(i)} cy={yFor(gPct)} r={isSelected ? 4.5 : 3} fill="#3b82f6" stroke="#fff" strokeWidth={1}>
                  <title>{`Grounding: ${gPct.toFixed(0)}%`}</title>
                </circle>
              )}
              {fPct !== null && (
                <rect
                  x={xFor(i) - (isSelected ? 4 : 2.6)}
                  y={yFor(fPct) - (isSelected ? 4 : 2.6)}
                  width={isSelected ? 8 : 5.2}
                  height={isSelected ? 8 : 5.2}
                  fill="#10b981"
                  stroke="#fff"
                  strokeWidth={1}
                  transform={`rotate(45 ${xFor(i)} ${yFor(fPct)})`}
                >
                  <title>{`Faithfulness: ${fPct.toFixed(0)}%`}</title>
                </rect>
              )}
              {isSelected && (
                <line x1={xFor(i)} y1={padT} x2={xFor(i)} y2={padT + innerH} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 3" />
              )}
            </g>
          );
        })}
      </svg>
      <div className="flex items-center gap-4 mt-2 text-[11px] text-gray-500">
        <span className="flex items-center gap-1.5">
          <svg width="14" height="8" className="flex-shrink-0"><line x1="0" y1="4" x2="14" y2="4" stroke="#3b82f6" strokeWidth="2" /></svg>
          Grounding rate (●)
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="14" height="8" className="flex-shrink-0"><line x1="0" y1="4" x2="14" y2="4" stroke="#10b981" strokeWidth="2" strokeDasharray="4 3" /></svg>
          Faithfulness rate (◆)
        </span>
        <span className="ml-auto text-gray-400">{fmtDate(reports[0].generated_at)} → {fmtDate(reports[n - 1].generated_at)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail panel — drill-down for one EvalReportRecord
// ---------------------------------------------------------------------------

function DetailPanel({ detail, loading }: { detail: EvalReportDetail | null; loading: boolean }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-400 gap-2">
        <Spinner /> Loading report…
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-400">
        Select a point on the chart or a row below to see grounding &amp; faithfulness detail.
      </div>
    );
  }

  const { grounding_results, faithfulness_results, uncited_sentences, completeness, relevance } = detail.report;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-gray-900 truncate">{detail.query}</p>
          <p className="text-xs text-gray-400 mt-0.5">{fmtDate(detail.generated_at)}</p>
        </div>
        <span className={`flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full ${
          detail.passed ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-600 border border-red-200'
        }`}>
          {detail.passed ? 'Passed' : 'Failed'}
        </span>
      </div>

      {detail.failure_reasons.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-700 space-y-0.5">
          {detail.failure_reasons.map((reason, i) => <p key={i}>{reason}</p>)}
        </div>
      )}

      {/* Relevance */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Relevance ({relevance.score}/5)
        </p>
        <div className="bg-white border border-gray-200 rounded-lg px-3 py-2">
          <p className="text-xs text-gray-700">{relevance.reasoning}</p>
        </div>
      </div>

      {/* Completeness */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Completeness ({fmtPct(completeness.recall_score * 100)} of subtopics covered)
        </p>
        <div className="space-y-1.5">
          {completeness.subtopics.length === 0 ? (
            <p className="text-xs text-gray-400">No subtopics generated for this query.</p>
          ) : completeness.subtopics.map((s, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {s.covered ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700">{s.subtopic}</p>
                {s.note && <p className="text-[10px] text-gray-400 mt-0.5">{s.note}</p>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Grounding results */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Citation Grounding ({grounding_results.length})
        </p>
        <div className="space-y-1.5">
          {grounding_results.length === 0 ? (
            <p className="text-xs text-gray-400">No findings to check.</p>
          ) : grounding_results.map((g, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {g.grounded ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700 line-clamp-2">{g.evidence_span}</p>
                <p className="text-[10px] text-gray-400 mt-0.5 truncate">
                  {g.method} · sim {g.similarity.toFixed(2)} · {g.citation_url}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Faithfulness verdicts */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Faithfulness Verdicts ({faithfulness_results.length})
        </p>
        <div className="space-y-1.5">
          {faithfulness_results.length === 0 ? (
            <p className="text-xs text-gray-400">No cited sentences to check.</p>
          ) : faithfulness_results.map((f, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {f.faithful ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700 line-clamp-2">[{f.citation_index}] {f.report_sentence}</p>
                <p className="text-[10px] text-gray-400 mt-0.5">{f.reasoning}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Uncited sentences */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Uncited Sentences ({uncited_sentences.length})
        </p>
        {uncited_sentences.length === 0 ? (
          <p className="text-xs text-gray-400">Every sentence in the report carries a citation.</p>
        ) : (
          <div className="space-y-1.5">
            {uncited_sentences.map((u, i) => (
              <div key={i} className="bg-white border border-gray-200 rounded-lg px-3 py-2">
                <p className="text-xs text-gray-700 line-clamp-2">{u.sentence}</p>
                <p className="text-[10px] text-gray-400 mt-0.5">{u.section}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------------

export default function EvalDashboard({ apiBase, clientId }: { apiBase: string; clientId: string }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [reports, setReports] = useState<EvalReportSummary[]>([]);
  const [globalSummary, setGlobalSummary] = useState<GlobalSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [runningEvalFor, setRunningEvalFor] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvalReportDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  // Model used to judge faithfulness/completeness/relevance for "Run Eval".
  const [evalModelOptions, setEvalModelOptions] = useState<ModelOption[]>([
    { id: 'gpt-5.4', label: 'GPT-5.4', description: 'Best for complex topics' },
    { id: 'gpt-5.4-mini', label: 'GPT-5.4 Mini', description: 'Faster and cheaper' },
  ]);
  const [evalModel, setEvalModel] = useState('gpt-5.4-mini');

  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then(res => res.json())
      .then((data: { default: string; options: ModelOption[] }) => {
        if (data.options?.length) setEvalModelOptions(data.options);
      })
      .catch(() => { /* keep the hardcoded fallback */ });
  }, [apiBase]);

  const headers = useMemo(() => ({ 'X-Client-Id': clientId }), [clientId]);

  const load = useCallback(async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const [runsRes, reportsRes, globalRes] = await Promise.all([
        fetch(`${apiBase}/runs?status=done`, { headers }),
        fetch(`${apiBase}/eval/reports`, { headers }),
        fetch(`${apiBase}/eval/summary`),
      ]);
      if (!runsRes.ok || !reportsRes.ok || !globalRes.ok) throw new Error('Failed to load eval data');
      setRuns(await runsRes.json());
      setReports(await reportsRes.json());
      setGlobalSummary(await globalRes.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase, headers]);

  useEffect(() => { load(); }, [load]);

  // Most recent eval report per run_id, for the runs table's status badge.
  const latestByRun = useMemo(() => {
    const map = new Map<string, EvalReportSummary>();
    for (const r of reports) {
      const existing = map.get(r.run_id);
      if (!existing || r.generated_at > existing.generated_at) map.set(r.run_id, r);
    }
    return map;
  }, [reports]);

  // Chronological order for the trend chart (oldest -> newest).
  const trendReports = useMemo(
    () => [...reports].sort((a, b) => a.generated_at.localeCompare(b.generated_at)),
    [reports],
  );

  const summary = useMemo(() => {
    if (reports.length === 0) return null;
    const passed = reports.filter(r => r.passed).length;
    const totalFindings = reports.reduce((s, r) => s + r.total_findings, 0);
    const ungrounded = reports.reduce((s, r) => s + r.ungrounded_count, 0);
    const totalCitations = reports.reduce((s, r) => s + r.total_citations, 0);
    const unfaithful = reports.reduce((s, r) => s + r.unfaithful_count, 0);
    const totalRecall = reports.reduce((s, r) => s + r.recall_score, 0);
    const totalRelevance = reports.reduce((s, r) => s + r.relevance_score, 0);
    const totalCost = reports.reduce((s, r) => s + r.eval_cost_usd, 0);
    return {
      runsEvaluated: reports.length,
      passRate: (passed / reports.length) * 100,
      groundingRate: totalFindings === 0 ? null : ((totalFindings - ungrounded) / totalFindings) * 100,
      faithfulnessRate: totalCitations === 0 ? null : ((totalCitations - unfaithful) / totalCitations) * 100,
      completenessRate: (totalRecall / reports.length) * 100,
      relevanceScore: totalRelevance / reports.length,
      totalCost,
    };
  }, [reports]);

  const loadDetail = useCallback(async (reportId: string) => {
    setSelectedReportId(reportId);
    setDetailLoading(true);
    try {
      const res = await fetch(`${apiBase}/eval/reports/${reportId}`, { headers });
      if (!res.ok) throw new Error('Failed to load report detail');
      setDetail(await res.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setDetailLoading(false);
    }
  }, [apiBase, headers]);

  async function runEval(runId: string) {
    setRunningEvalFor(runId);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/runs/${runId}/eval?model=${encodeURIComponent(evalModel)}`, {
        method: 'POST',
        headers,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `HTTP ${res.status}`);
      }
      const record: EvalReportDetail = await res.json();
      setDetail(record);
      setSelectedReportId(record.id);
      await load();
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setRunningEvalFor(null);
    }
  }

  async function clearAll() {
    if (!window.confirm('Delete all eval history? This cannot be undone.')) return;
    setClearing(true);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/eval/reports`, { method: 'DELETE', headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSelectedReportId(null);
      setDetail(null);
      await load();
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto">
    <div className="p-4 sm:p-6 space-y-6 max-w-5xl mx-auto w-full">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Eval Dashboard</h2>
          <p className="text-sm text-gray-400 mt-0.5">
            Citation grounding &amp; faithfulness checks across completed research runs.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs font-medium text-gray-500">
            Eval model:
            <select
              value={evalModel}
              onChange={e => setEvalModel(e.target.value)}
              className="border border-gray-200 rounded-lg px-2 py-1.5 text-xs font-medium text-gray-700 bg-white focus:outline-none"
            >
              {evalModelOptions.map(o => (
                <option key={o.id} value={o.id}>{o.label ?? o.id}</option>
              ))}
            </select>
          </label>
          <button
            onClick={clearAll}
            disabled={clearing || reports.length === 0}
            className="text-xs font-semibold text-red-600 hover:text-red-700 disabled:opacity-40 transition-colors px-2.5 py-1.5 rounded-lg border border-red-200 hover:bg-red-50"
          >
            {clearing ? 'Clearing…' : 'Clear All'}
          </button>
        </div>
      </div>

      {errorMsg && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2">
          {errorMsg}
        </p>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-32 text-sm text-gray-400 gap-2">
          <Spinner /> Loading…
        </div>
      ) : (
        <>
          {/* Your stats */}
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-2">Your Stats</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Runs Evaluated</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary?.runsEvaluated ?? 0}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Pass Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.passRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Grounding Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.groundingRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Faithfulness Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.faithfulnessRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Completeness</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.completenessRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Relevance</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtScore(summary.relevanceScore) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Eval Cost</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtCost(summary.totalCost) : '—'}</p>
              </div>
            </div>
          </div>

          {/* Community average — aggregated across all visitors, read-only */}
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-2">
              Community Average
              <span className="normal-case text-gray-400 font-normal">
                {' '}— across all visitors
                {globalSummary && globalSummary.runs_evaluated > 0
                  ? ` (${globalSummary.runs_evaluated.toLocaleString()} eval${globalSummary.runs_evaluated !== 1 ? 's' : ''})`
                  : ''}
              </span>
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Pass Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(globalSummary?.pass_rate ?? null)}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Grounding Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(globalSummary?.grounding_rate ?? null)}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Faithfulness Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(globalSummary?.faithfulness_rate ?? null)}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Completeness</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(globalSummary?.completeness_rate ?? null)}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Relevance</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{fmtScore(globalSummary?.relevance_score ?? null)}</p>
              </div>
            </div>
          </div>

          {/* Trend chart */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-2">Quality Over Time</p>
            <TrendChart reports={trendReports} selectedId={selectedReportId} onSelect={loadDetail} />
          </div>

          {/* Runs table */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
            <p className="text-sm font-semibold text-gray-900 px-4 pt-4 pb-2">Recent Research</p>
            {runs.length === 0 ? (
              <p className="text-sm text-gray-400 px-4 pb-4">No completed research runs yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold border-b border-gray-200">
                      <th className="text-left px-4 py-2 font-semibold">Query</th>
                      <th className="text-left px-4 py-2 font-semibold">Started</th>
                      <th className="text-right px-4 py-2 font-semibold">Cost</th>
                      <th className="text-right px-4 py-2 font-semibold">Tokens</th>
                      <th className="text-right px-4 py-2 font-semibold">Elapsed</th>
                      <th className="text-left px-4 py-2 font-semibold">Eval</th>
                      <th className="px-4 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map(run => {
                      const latest = latestByRun.get(run.id);
                      const isRunning = runningEvalFor === run.id;
                      return (
                        <tr key={run.id} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
                          <td className="px-4 py-2.5 max-w-[260px]">
                            <p className="truncate text-gray-800">{run.query || 'Untitled research'}</p>
                          </td>
                          <td className="px-4 py-2.5 text-gray-500 whitespace-nowrap">{fmtDate(run.started_at)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{fmtCost(run.stats?.cost_usd)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.total_tokens?.toLocaleString() ?? '—'}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.elapsed_seconds !== undefined ? `${run.stats.elapsed_seconds}s` : '—'}</td>
                          <td className="px-4 py-2.5 whitespace-nowrap">
                            {latest ? (
                              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                                latest.passed ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-600 border border-red-200'
                              }`}>
                                {latest.passed ? 'Passed' : 'Failed'}
                              </span>
                            ) : (
                              <span className="text-xs text-gray-400">Not evaluated</span>
                            )}
                          </td>
                          <td className="px-4 py-2.5 text-right whitespace-nowrap">
                            {latest ? (
                              <button
                                onClick={() => loadDetail(latest.id)}
                                className="text-xs font-semibold text-blue-600 hover:text-blue-800 transition-colors"
                              >
                                View
                              </button>
                            ) : (
                              <button
                                onClick={() => runEval(run.id)}
                                disabled={isRunning}
                                className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-800 disabled:opacity-50 transition-colors"
                              >
                                {isRunning && <Spinner />} {isRunning ? 'Running…' : 'Run Eval'}
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Detail panel */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-3">Report Detail</p>
            <DetailPanel detail={detail} loading={detailLoading} />
          </div>
        </>
      )}
    </div>
    </div>
  );
}
