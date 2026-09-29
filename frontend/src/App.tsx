import { useEffect, useMemo, useState } from "react";
import "./site-navigation.css";
import "./site-navigation.js";
import * as api from "./api";
import { OrderCard, PageCard, termsOf } from "./Cards";
import { save, toMarkdown } from "./download";
import type { Access, Addendum, Filters, OrderHit, PageHit, Results, SpiderSettings, Status } from "./types";

const PARTS = [3, 4, 5, 6, 7, 8, 9];
const PART_YEARS: Record<number, string> = { 3: "1751–81", 4: "1782–87", 5: "1788–1806", 6: "1807–10", 7: "1811–19", 8: "1819–53", 9: "1853–85" };
const VOLS = [1, 2, 3];
const DEFAULT_SETTINGS: SpiderSettings = location.pathname.startsWith("/searchspider")
  ? { reading_rounds: 0 } : { reading_rounds: 1 };
const PAGE_SIZE = 20;

function loadSettings(): SpiderSettings {
  try {
    const saved = JSON.parse(localStorage.getItem("spiderSettings") ?? "{}");
    const rounds = saved.reading_rounds ?? saved.recursive_searches;
    return { reading_rounds: Number.isInteger(rounds)
        ? Math.max(0, Math.min(4, rounds)) : DEFAULT_SETTINGS.reading_rounds };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

const statusOf = (x: OrderHit | PageHit) => x.status ?? (x.spider?.v === "r" ? "relevant" : x.spider?.v === "p" ? "partial" : x.spider ? "dropped" : "not_assessed");

const EMPTY: Filters = { orders: true, chronicle: true, yearFrom: "", yearTo: "", parts: [], volumes: [] };

function toggle(list: number[], n: number) {
  return list.includes(n) ? list.filter((x) => x !== n) : [...list, n].sort();
}

const tok = (i: number, o: number) => `${i.toLocaleString()} in / ${o.toLocaleString()} out tokens`;

function Cited({ text, r, onCite }: { text: string; r: Results; onCite: (alias: string) => void }) {
  return (
    <div className="summary">
      {(text || "No answer.").split(/\n\s*\n/).map((para, pi) => (
        <p key={pi}>
          {para.split(/(\[[A-Z]\d+\])/g).map((s, i) => {
            const m = s.match(/^\[([A-Z]\d+)\]$/);
            if (!m) return <span key={i}>{s}</span>;
            const ref = r.refs?.[m[1]];
            return (
              <a key={i} className="cite-chip" href={`#card-${m[1]}`} title={ref?.label}
                onClick={(e) => { e.preventDefault(); onCite(m[1]); }}>
                {m[1]}
              </a>
            );
          })}
        </p>
      ))}
    </div>
  );
}

type Busy = "" | "deeper" | "followup";

function Continue({ r, busy, onDeeper, onFollowup }: { r: Results; busy: Busy; onDeeper: () => void; onFollowup: (q: string) => Promise<boolean> }) {
  const [fq, setFq] = useState("");
  const left = r.counts?.not_assessed ?? 0;
  if (!r.run_id) return null;
  return (
    <section className="continuepanel" aria-label="Continue this research">
      <h3>Continue this research</h3>
      <div className="continuegrid">
        <div className="continuebox">
          <button type="button" className="primary" disabled={!!busy || left === 0} onClick={onDeeper}>
            {busy === "deeper" ? "Reading deeper…" : "Push deeper"}
          </button>
          <span className="meta">{left === 0 ? "Every gathered document has been judged." : `Judge the next ${Math.min(50, left)} of ${left.toLocaleString()} unjudged documents`}</span>
        </div>
        <form className="continuebox followup" onSubmit={async (e) => { e.preventDefault(); if (fq.trim() && await onFollowup(fq.trim())) setFq(""); }}>
          <input value={fq} onChange={(e) => setFq(e.target.value)} placeholder="Ask a follow-up about this evidence" disabled={!!busy} aria-label="Follow-up question" />
          <button type="submit" className="primary" disabled={!!busy || !fq.trim()}>{busy === "followup" ? "Answering…" : "Ask follow-up"}</button>
        </form>
      </div>
    </section>
  );
}

function Trail({ r }: { r: Results }) {
  const rounds = [...(r.rounds ?? [])].sort((a, b) => (a.step === "final" ? 1e9 : a.step) - (b.step === "final" ? 1e9 : b.step));
  if (!rounds.length) return null;
  return (
    <details className="trail">
      <summary>Research trail ({rounds.length} stages)</summary>
      <div className="rounds">
        {rounds.map((rd, i) => (
          <div key={i} className="round">
            <div className="roundhead">
              {rd.label ?? `Stage ${rd.step}`}
              {rd.in > 0 && <span className="meta"> · {tok(rd.in, rd.out)}</span>}
            </div>
            {rd.text && <div className="modeltext">{rd.text}</div>}
          </div>
        ))}
        {r.aliases && r.aliases.length > 0 && (
          <div className="round">
            <div className="roundhead">Name variants also searched</div>
            <div className="modeltext">{r.aliases.join(", ")}</div>
          </div>
        )}
      </div>
    </details>
  );
}

function Summary({ r, onCite, busy, onDeeper, onFollowup }: { r: Results; onCite: (alias: string) => void; busy: Busy; onDeeper: () => void; onFollowup: (q: string) => Promise<boolean> }) {
  return (
    <>
      <section className="spiderpanel">
        <div className="spiderhead">
          <span className="spiderlogo">SearchSpider</span>
          <span className="meta">
            {r.usage && `${r.usage.calls} calls · ${tok(r.usage.in, r.usage.out)} · `}
            {(r.ms / 1000).toFixed(1)} s
          </span>
        </div>
        {r.counts && (
          <div className="counts">
            Gathered {(r.orders.length + r.pages.length).toLocaleString()} documents ({r.orders.length.toLocaleString()} orders, {r.pages.length.toLocaleString()} chronicle pages).
            {" "}Judged {(r.counts.relevant + r.counts.partial + r.counts.dropped + r.counts.missing).toLocaleString()}: <b>{r.counts.relevant} relevant</b>, <b>{r.counts.partial} partly relevant</b>, {r.counts.dropped} dropped{r.counts.missing ? `, ${r.counts.missing} not returned` : ""}.
            {" "}{r.counts.not_assessed.toLocaleString()} unjudged.
          </div>
        )}
        <Cited text={r.summary ?? ""} r={r} onCite={onCite} />
        {(r.addenda ?? []).map((a) => (
          <div key={`${a.kind}${a.n}`} className={`addendum ${a.kind}`}>
            <div className="addhead">
              <span>{a.kind === "followup" ? <>Follow-up {a.n}: <b>{a.q}</b></> : <>Push deeper {a.n}: <b>{a.assessed} more documents judged</b> — {a.tally}</>}</span>
              {a.in !== undefined && <span className="meta">{tok(a.in, a.out ?? 0)}</span>}
            </div>
            <Cited text={a.text} r={r} onCite={onCite} />
          </div>
        ))}
      </section>
      <Continue r={r} busy={busy} onDeeper={onDeeper} onFollowup={onFollowup} />
      <Trail r={r} />
    </>
  );
}

function Slider({ label, value, min, max, step, fmt, onChange, disabled = false }: { label: string; value: number; min: number; max: number; step: number; fmt: (n: number) => string; onChange: (n: number) => void; disabled?: boolean }) {
  return (
    <label className="slider">
      <span>{label}</span>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
      <b>{fmt(value)}</b>
    </label>
  );
}

function Pagination({ page, count, onChange, label }: { page: number; count: number; onChange: (n: number) => void; label: string }) {
  const total = Math.ceil(count / PAGE_SIZE);
  if (total < 2) return null;
  return <nav className="pagination" aria-label={`${label} result pages`}>
    <span>{(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, count)} of {count}</span>
    <button disabled={page === 1} onClick={() => onChange(page - 1)}>Previous</button>
    <div className="page-numbers">{Array.from({ length: total }, (_, i) => i + 1).map(n =>
      <button key={n} aria-current={n === page ? "page" : undefined} onClick={() => onChange(n)}>{n}</button>
    )}</div>
    <button disabled={page === total} onClick={() => onChange(page + 1)}>Next</button>
  </nav>;
}

export default function App() {
  // A refresh starts blank: the search is not kept in the address bar.
  const [q, setQ] = useState("");
  const [f, setF] = useState<Filters>(EMPTY);
  const [useSpider, setUseSpider] = useState(false);
  const [st, setSt] = useState<Status | null>(null);
  const [access, setAccess] = useState<Access | null>(null);
  const [disconnecting, setDisconnecting] = useState(false);
  const [res, setRes] = useState<Results | null>(null);
  const [asked, setAsked] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [settings, setSettings] = useState<SpiderSettings>(loadSettings);
  const [chrono, setChrono] = useState(false);
  const [showRelevant, setShowRelevant] = useState(true);
  const [showPartial, setShowPartial] = useState(true);
  const [showDropped, setShowDropped] = useState(false);
  const [cutoff, setCutoff] = useState(0);
  const [kw, setKw] = useState<string[]>([]); // selected keyword stems: results must contain all of them
  const [resKey, setResKey] = useState(0); // changes only when a new search replaces the results
  const [extraBusy, setExtraBusy] = useState<"" | "deeper" | "followup">("");
  const [showUnassessed, setShowUnassessed] = useState(false);
  const [orderPage, setOrderPage] = useState(1);
  const [chroniclePage, setChroniclePage] = useState(1);
  const [pendingAlias, setPendingAlias] = useState<string | null>(null);
  const depthEnabled = access?.mode === "paid" || access?.mode === "local";
  const effectiveRounds = depthEnabled ? settings.reading_rounds : 0;

  // Appended follow-ups and deeper tranches update res in place and keep the current page.
  useEffect(() => { setOrderPage(1); setChroniclePage(1); }, [resKey, chrono, showRelevant, showPartial, showDropped, showUnassessed, cutoff, kw]);
  useEffect(() => {
    if (!pendingAlias) return;
    const frame = requestAnimationFrame(() => {
      document.getElementById(`card-${pendingAlias}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      setPendingAlias(null);
    });
    return () => cancelAnimationFrame(frame);
  }, [pendingAlias, orderPage, chroniclePage]);

  function setSetting(k: keyof SpiderSettings, v: number) {
    const next = { ...settings, [k]: v };
    setSettings(next);
    try { localStorage.setItem("spiderSettings", JSON.stringify(next)); } catch { /* storage unavailable */ }
  }

  useEffect(() => {
    if (location.hash) history.replaceState(null, "", location.pathname + location.search);
    api.status().then(setSt).catch(() => setErr("Search server not reachable. Please try again shortly."));
    api.access().then(setAccess).catch(() => setErr("Unable to check AI access. Basic search is still available."));
  }, []);

  function refreshAccess() { api.access().then(setAccess).catch(() => {}); }

  async function disconnectAccount() {
    setDisconnecting(true);
    try {
      await api.logout();
      setAccess((current) => current ? { ...current, mode: "free", connected: false } : current);
      refreshAccess();
    } catch (e) {
      setErr(`Unable to disconnect: ${(e as Error).message}`);
    } finally {
      setDisconnecting(false);
    }
  }

  async function run(query = q, filters = f, spiderOn = useSpider) {
    if (!query.trim()) return;
    setBusy(true);
    setErr("");
    try {
      setRes(spiderOn ? await api.spider(query, filters, { reading_rounds: effectiveRounds }) : await api.search(query, filters));
      setAsked(query);
      setCutoff(0);
      setKw([]);
      setResKey((k) => k + 1);
    } catch (e) {
      setErr(String((e as Error).message));
    } finally {
      setBusy(false);
      refreshAccess();
    }
  }

  async function pushDeeper() {
    if (!res?.run_id || extraBusy) return;
    setExtraBusy("deeper");
    setErr("");
    try {
      const d = await api.deeper(res.run_id);
      const upd = new Map(d.updated.map((x) => [x.alias, x]));
      setRes((prev) => prev && {
        ...prev,
        // same items in the same order; only newly judged ones change
        orders: prev.orders.map((o) => (upd.get(o.alias) as OrderHit | undefined) ?? o),
        pages: prev.pages.map((p) => (upd.get(p.alias) as PageHit | undefined) ?? p),
        counts: d.counts, trail: d.trail, rounds: d.rounds, usage: d.usage,
        addenda: [...(prev.addenda ?? []), { kind: "deeper", n: d.deeper.n, text: d.deeper.text, tally: d.deeper.tally, assessed: d.deeper.assessed, in: d.deeper.in, out: d.deeper.out } as Addendum],
      });
    } catch (e) {
      setErr(String((e as Error).message));
    } finally {
      setExtraBusy("");
      refreshAccess();
    }
  }

  async function askFollowup(fq: string): Promise<boolean> {
    if (!res?.run_id || extraBusy) return false;
    setExtraBusy("followup");
    setErr("");
    try {
      const d = await api.followup(res.run_id, fq);
      setRes((prev) => prev && {
        ...prev, trail: d.trail, rounds: d.rounds, usage: d.usage,
        addenda: [...(prev.addenda ?? []), { kind: "followup", n: d.followup.n, q: d.followup.q, text: d.followup.text, in: d.followup.in, out: d.followup.out } as Addendum],
      });
      return true;
    } catch (e) {
      setErr(String((e as Error).message));
      return false;
    } finally {
      setExtraBusy("");
      refreshAccess();
    }
  }

  const terms = useMemo(() => res?.highlight_terms ?? termsOf(asked, ...(res?.aliases ?? []), ...((res?.trail ?? []).filter((t) => t.tool === "search").map((t) => String(t.args.query ?? "")))), [asked, res]);
  const isSpider = res?.summary !== undefined;
  const visible = (x: OrderHit | PageHit) => {
    if (kw.length && !kw.every((t) => x.terms?.includes(t))) return false;
    const st = statusOf(x);
    if (x.score < cutoff && !(isSpider && (st === "relevant" || st === "partial"))) return false;
    if (!isSpider) return true;
    return st === "relevant" ? showRelevant : st === "partial" ? showPartial : st === "dropped" ? showDropped : showUnassessed;
  };
  const maxScore = useMemo(() => Math.ceil(100 * Math.max(0, ...[...(res?.orders ?? []), ...(res?.pages ?? [])].map((x) => x.score))) / 100, [res]);
  const hiddenByCutoff = useMemo(() => [...(res?.orders ?? []), ...(res?.pages ?? [])]
    .filter((x) => x.score < cutoff && !(isSpider && ["relevant", "partial"].includes(statusOf(x)))).length, [res, cutoff, isSpider]);
  const orders = useMemo(() => {
    const o = (res?.orders ?? []).filter(visible);
    if (chrono) return [...o].sort((a, b) => a.date.localeCompare(b.date));
    return o; // server preserves verdict groups and hybrid relevance order
  }, [res, chrono, showRelevant, showPartial, showDropped, showUnassessed, cutoff, kw, isSpider]);
  const pages = useMemo(() => {
    const p = (res?.pages ?? []).filter(visible);
    if (chrono) return [...p].sort((a, b) => a.volume - b.volume || a.page - b.page);
    return p;
  }, [res, chrono, showRelevant, showPartial, showDropped, showUnassessed, cutoff, kw, isSpider]);
  function goToCitation(alias: string) {
    const oi = orders.findIndex(o => o.alias === alias);
    const pi = pages.findIndex(p => p.alias === alias);
    if (oi >= 0) setOrderPage(Math.floor(oi / PAGE_SIZE) + 1);
    if (pi >= 0) setChroniclePage(Math.floor(pi / PAGE_SIZE) + 1);
    setPendingAlias(alias);
  }
  const spiderOk = st?.spider ?? false;

  return (
    <>
    <nav className="bnr-navigation" aria-label="Primary navigation">
      <div className="bnr-navigation__links"><a data-site-section="reader" href="/reader">Neural Reader</a>
      <details><summary>Konbaung Digital History Project</summary><div className="bnr-navigation__menu">
        <a data-site-section="chronicles" href="/chronicles">Chronicle Reader</a>
        <a data-site-section="knowledge-graph" href="/knowledge-graph">Knowledge Graph</a>
        <a data-site-section="api" href="/api/v1/docs">Graph API</a>
        <a data-site-section="searchspider" href="/searchspider/" aria-current="page">SearchSpider</a>
      </div></details><a href="https://language-engine.ai/">Language Engine</a></div>
    </nav>
    <div className="app">
      <header className="top">
        <h1>SearchSpider</h1>
        <p className="sub">Agentic retrieval for Konbaung historical documents</p>
      </header>

      {access && access.mode !== "local" && <section className="access-panel" aria-label="AI access">
        <div><strong>{access.mode === "paid" ? "Language Engine Pro connected" : access.connected ? "Language Engine account connected" : "Free community allowance"}</strong>
          <p>{access.mode === "paid" ? "Searches and follow-ups use your existing Language Engine token budget."
            : <>Try agentic searches, follow-ups and deeper reading from a shared daily budget. Subscribe to our parent service, <a href="https://language-engine.ai/?signup=pro">Language Engine Pro</a>, for ongoing access.</>}</p>
        </div>
        <div className="access-links">
          {access.connected
            ? <button type="button" className="link" disabled={disconnecting} onClick={disconnectAccount}>{disconnecting ? "Disconnecting…" : "Disconnect"}</button>
            : <a href={access.login_url}>Connect Language Engine account</a>}
        </div>
      </section>}

      <form className="searchbar" onSubmit={(e) => { e.preventDefault(); run(); }}>
        <input value={q} maxLength={2000} aria-label="Search historical documents" onChange={(e) => setQ(e.target.value)} placeholder="e.g. ritual processions in which Alaungpaya took part" autoFocus />
        <button type="submit" disabled={busy}>{busy ? (useSpider ? "spider working…" : "searching…") : "Search"}</button>
        <button type="button" role="switch" aria-checked={useSpider} aria-label="Use SearchSpider agentic search"
          disabled={!spiderOk || busy} onClick={() => setUseSpider(!useSpider)}
          className={`spidertoggle ${useSpider ? "on" : ""} ${spiderOk ? "" : "off"}`} title={spiderOk ? `Research agent (${st?.spider_model}): searches, reads ranked evidence, judges and summarises` : st?.spider_error ?? "unavailable"}>
          <span className="knob" aria-hidden="true" /> SearchSpider
        </button>
      </form>

      {useSpider && spiderOk && (
        <div className="spidercontrols">
          <Slider label="Additional reading rounds" value={effectiveRounds} min={0} max={4} step={1} disabled={!depthEnabled || busy} fmt={(n) => n === 0 ? "Off" : `${n} round${n === 1 ? "" : "s"}`} onChange={(n) => setSetting("reading_rounds", n)} />
          <button type="button" className="link" onClick={() => { setSettings(DEFAULT_SETTINGS); try { localStorage.removeItem("spiderSettings"); } catch { /* */ } }}>defaults</button>
        </div>
      )}

      <div className="filters">
        <label><input type="checkbox" checked={f.orders} onChange={(e) => setF({ ...f, orders: e.target.checked })} /> Royal Orders</label>
        <label><input type="checkbox" checked={f.chronicle} onChange={(e) => setF({ ...f, chronicle: e.target.checked })} /> Chronicle</label>
        <span className="sep" />
        <label title="Applies to Royal Orders (the chronicle is not dated per sentence)">
          Royal Orders years <input className="yr" value={f.yearFrom} onChange={(e) => setF({ ...f, yearFrom: e.target.value.replace(/\D/g, "") })} placeholder="1751" />
          – <input className="yr" value={f.yearTo} onChange={(e) => setF({ ...f, yearTo: e.target.value.replace(/\D/g, "") })} placeholder="1885" />
        </label>
        <span className="sep" />
        <span className="group" title="Than Tun, Royal Orders of Burma: Part and years">Royal Orders Parts {PARTS.map((p) => (
          <button key={p} type="button" title={PART_YEARS[p]} className={`pill ${f.parts.includes(p) ? "on" : ""}`} onClick={() => setF({ ...f, parts: toggle(f.parts, p) })}>{p}</button>
        ))}</span>
        <span className="group" title="Konbaungset Yazawin volume">Chronicle vols {VOLS.map((v) => (
          <button key={v} type="button" className={`pill ${f.volumes.includes(v) ? "on" : ""}`} onClick={() => setF({ ...f, volumes: toggle(f.volumes, v) })}>{v}</button>
        ))}</span>
        <span className="sep" />
        {JSON.stringify(f) !== JSON.stringify(EMPTY) && <button type="button" className="link" onClick={() => setF(EMPTY)}>reset</button>}
      </div>

      {res && (
        <div className="viewbar">
          <label title="Orders by date; chronicle pages in volume and page order"><input type="checkbox" checked={chrono} onChange={(e) => setChrono(e.target.checked)} /> Chronological</label>
          {isSpider && <label title="Items Gemini judged to bear directly on the question"><input type="checkbox" checked={showRelevant} onChange={(e) => setShowRelevant(e.target.checked)} /> Show relevant ({res?.counts?.relevant ?? 0})</label>}
          {isSpider && <label title="Items Gemini judged to bear on the question in part or in passing"><input type="checkbox" checked={showPartial} onChange={(e) => setShowPartial(e.target.checked)} /> Show partly relevant ({res?.counts?.partial ?? 0})</label>}
          {isSpider && <label title="Items Gemini judged to have nothing on the question"><input type="checkbox" checked={showDropped} onChange={(e) => setShowDropped(e.target.checked)} /> Show dropped ({res?.counts?.dropped ?? 0})</label>}
          {isSpider && <label title="All unjudged evidence, ordered by keyword and semantic relevance across the planned searches"><input type="checkbox" checked={showUnassessed} onChange={(e) => setShowUnassessed(e.target.checked)} /> Show unjudged ({(res?.counts?.not_assessed ?? 0) + (res?.counts?.missing ?? 0)})</label>}
          <span className="sep" />
          {maxScore > 0 && <label className="slider cutoff" title="Hide documents whose relevance score is below this value. Documents judged relevant or partly relevant are never hidden.">
            <span>Relevance cutoff</span>
            <input type="range" aria-label="Relevance cutoff" min={0} max={maxScore} step={0.01} value={Math.min(cutoff, maxScore)} onChange={(e) => setCutoff(Math.round(Number(e.target.value) * 100) / 100)} />
            <b>{cutoff > 0 ? `≥ ${cutoff.toFixed(2)} (${hiddenByCutoff} hidden)` : "off"}</b>
          </label>}
          <span className="sep" />
          <button className="link" title="Markdown file of what is currently shown: the answer, follow-ups, research trail, and the full text of every visible order and chronicle page in the current order" onClick={() => {
            const shown = [
              chrono ? "chronological order" : "relevance order",
              isSpider && `verdicts shown: ${[showRelevant && "relevant", showPartial && "partly relevant", showDropped && "dropped", showUnassessed && "unjudged"].filter(Boolean).join(", ") || "none"}`,
              kw.length > 0 && `containing all of: ${kw.map((t) => res?.keywords?.find((k) => k.stem === t)?.word ?? t).join(", ")}`,
              cutoff > 0 && `relevance cutoff ≥ ${cutoff.toFixed(2)}${isSpider ? " (not applied to relevant or partly relevant)" : ""}`,
              !f.orders && "Royal Orders hidden", !f.chronicle && "chronicle hidden",
            ].filter(Boolean).join("; ");
            save(`${asked.slice(0, 60).replace(/[^\w]+/g, "_")}.md`,
              toMarkdown(asked, res, isSpider ? settings : null, { orders: f.orders ? orders : [], pages: f.chronicle ? pages : [], chrono, description: shown }), "text/markdown");
          }}>Download results</button>
        </div>
      )}

      {res && (res.keywords?.length ?? 0) > 0 && (
        <div className="keywordbar" aria-label="Narrow by keyword">
          <span>Keywords</span>
          {[...res.keywords!].sort((a, b) => a.word.localeCompare(b.word)).map((k) => (
            <button key={k.stem} type="button" className={`pill ${kw.includes(k.stem) ? "on" : ""}`} aria-pressed={kw.includes(k.stem)}
              title={`${k.n.toLocaleString()} documents contain this word or its variants`}
              onClick={() => setKw(kw.includes(k.stem) ? kw.filter((t) => t !== k.stem) : [...kw, k.stem])}>
              {k.word} <span className="n">{k.n.toLocaleString()}</span>
            </button>
          ))}
          {kw.length > 0 && <button type="button" className="link" onClick={() => setKw([])}>clear</button>}
        </div>
      )}

      {err && <div className="error">{err}</div>}
      {res?.summary !== undefined && <Summary r={res} onCite={goToCitation} busy={extraBusy} onDeeper={pushDeeper} onFollowup={askFollowup} />}


      {res && (
        <main className="results">
          {f.orders && (
            <section>
              <h2>Royal Orders <span className="count">{orders.length}</span></h2>
              {orders.length === 0 && <p className="empty">No orders matched.</p>}
              <Pagination label="Royal Orders" page={orderPage} count={orders.length} onChange={setOrderPage} />
              {orders.slice((orderPage - 1) * PAGE_SIZE, orderPage * PAGE_SIZE).map((o) => <OrderCard key={o.id} o={o} terms={terms} showMy />)}
              <Pagination label="Royal Orders" page={orderPage} count={orders.length} onChange={setOrderPage} />
            </section>
          )}
          {f.chronicle && (
            <section>
              <h2>Konbaungset Yazawin <span className="count">{pages.length} pages</span></h2>
              {pages.length === 0 && <p className="empty">No chronicle pages matched.</p>}
              <Pagination label="Chronicle" page={chroniclePage} count={pages.length} onChange={setChroniclePage} />
              {pages.slice((chroniclePage - 1) * PAGE_SIZE, chroniclePage * PAGE_SIZE).map((p) => <PageCard key={p.page_id} p={p} terms={terms} showMy />)}
              <Pagination label="Chronicle" page={chroniclePage} count={pages.length} onChange={setChroniclePage} />
            </section>
          )}
        </main>
      )}
    </div>
    </>
  );
}
