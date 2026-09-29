import { useState } from "react";
import type { OrderHit, PageHit, Verdict } from "./types";

// The backend supplies corpus word forms matched by the shared stemmer. Fallback is literal.
export function termsOf(...texts: string[]): string[] {
  return [...new Set(texts.flatMap(s => s.toLowerCase().match(/[a-z0-9]+/g) ?? []))];
}

function Marked({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length) return <>{text}</>;
  const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  const out: (string | JSX.Element)[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    out.push(text.slice(last, m.index), <mark key={m.index}>{m[0]}</mark>);
    last = (m.index ?? 0) + m[0].length;
  }
  out.push(text.slice(last));
  return <>{out}</>;
}

function SpiderBadge({ v, judged, status, score }: { v?: Verdict; judged?: boolean; status?: string; score?: number | null }) {
  if (v === undefined && !status) return null;
  const st = status ?? (v === null ? (judged === false ? "not_assessed" : "dropped") : v?.v === "r" ? "relevant" : v?.v === "p" ? "partial" : "dropped");
  const label: Record<string, string> = { relevant: "relevant", partial: "partly relevant", dropped: "dropped", missing: "missing verdict", not_assessed: "not assessed" };
  const cls: Record<string, string> = { relevant: "rel", partial: "part" };
  return (
    <>
      {score != null && <span className="found" title="Keyword and semantic relevance across the search queries">relevance {score.toFixed(3)}</span>}
      <span className={`badge ${cls[st] ?? "dim"}`}>{label[st]}</span>
      {v?.s && <p className="note">{v.s}</p>}
    </>
  );
}

function Found({ found, read }: { found?: string[]; read?: boolean }) {
  if (!found?.length) return null;
  const kinds = [...new Set(found.map((f) => (f.startsWith("graph") ? "graph" : f.startsWith("search") ? "search" : f.split(":")[0])))];
  return <span className="found" title={found.join("\n")}>{read ? "read · " : ""}via {kinds.join(" + ")}</span>;
}

function Passages({ items, hits, terms, lang, open }: { items: string[]; hits: number[]; terms: string[]; lang: "en" | "my"; open: boolean }) {
  const hitSet = new Set(hits);
  const shown = open ? items.map((_, i) => i) : hits.length ? hits : items.length ? [0] : [];
  return (
    <div className={`passages ${lang}`} lang={lang === "my" ? "my" : "en"}>
      {shown.map((i, k) => (
        <p key={i} className={hitSet.has(i) ? "hit" : ""}>
          {!open && k > 0 && shown[k - 1] !== i - 1 && <span className="gap">… </span>}
          {lang === "en" ? <Marked text={items[i]} terms={terms} /> : items[i]}
        </p>
      ))}
    </div>
  );
}

export function OrderCard({ o, terms, showMy }: { o: OrderHit; terms: string[]; showMy: boolean }) {
  const [open, setOpen] = useState(false);
  const date = o.date_supplied ? `(${o.date})` : o.date;
  return (
    <article className="card" id={`card-${o.alias ?? o.id}`}>
      <header>
        {o.alias && <span className="alias">{o.alias}</span>}
        <strong>{date}</strong>
        {o.date_uncertain && <span className="flag" title="Year garbled in OCR; inferred from neighbouring orders">year uncertain</span>}
        <span className="cite">ROB Part {o.part}, PDF p. {o.pdf_pages[0]}{o.pdf_pages.length > 1 ? `–${o.pdf_pages[o.pdf_pages.length - 1]}` : ""}</span>
        <Found found={o.found} read={o.read} />
        {o.deeper && <span className="newchip" title="Judged by a push-deeper tranche after the answer was written">push deeper {o.deeper}</span>}
        <SpiderBadge v={o.spider} judged={o.judged} status={o.status} score={o.score} />
      </header>
      <Passages items={o.en_passages} hits={o.hit_en} terms={terms} lang="en" open={open} />
      {showMy && o.my_passages.length > 0 && (
        <>
          <div className="mylabel">Burmese {o.my_alignment === "same_date" && <span className="flag" title="Burmese and English order counts differ for this date; all Burmese orders of the date are shown">all orders of this date</span>}</div>
          <Passages items={o.my_passages} hits={o.hit_my} terms={[]} lang="my" open={open} />
        </>
      )}
      <footer>
        {o.topics.length > 0 && <span className="topics">{o.topics.slice(0, 6).join(" · ")}</span>}
        <button className="link" onClick={() => setOpen(!open)}>{open ? "matches only" : `full order (${o.en_passages.length} passages)`}</button>
      </footer>
    </article>
  );
}

export function PageCard({ p, terms, showMy }: { p: PageHit; terms: string[]; showMy: boolean }) {
  const [open, setOpen] = useState(false);
  const hits = new Set(p.hits);
  const shown = open ? p.sentences : p.sentences.filter((s) => hits.has(s.id));
  return (
    <article className="card" id={`card-${p.alias ?? p.page_id}`}>
      <header>
        {p.alias && <span className="alias">{p.alias}</span>}
        <strong>Yazawin vol. {p.volume}, p. {p.page}</strong>
        <span className="cite">{p.hits.length} matching sentence{p.hits.length > 1 ? "s" : ""}</span>
        <Found found={p.found} read={p.read} />
        {p.deeper && <span className="newchip" title="Judged by a push-deeper tranche after the answer was written">push deeper {p.deeper}</span>}
        <SpiderBadge v={p.spider} judged={p.judged} status={p.status} score={p.score} />
      </header>
      {shown.map((s) => (
        <div key={s.id} className={`sent ${hits.has(s.id) ? "hit" : ""}`}>
          {showMy && <p lang="my" className="my">{s.my}</p>}
          <p className="en"><Marked text={s.en} terms={hits.has(s.id) ? terms : []} /></p>
        </div>
      ))}
      <footer>
        <span className="topics">{p.sentences.length} sentences on this page</span>
        <button className="link" onClick={() => setOpen(!open)}>{open ? "matches only" : "full page"}</button>
      </footer>
    </article>
  );
}
